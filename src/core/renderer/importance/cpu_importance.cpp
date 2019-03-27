#include "cpu_importance.hpp"
#include "imp_decimater.hpp"
#include "imp_common.hpp"
#include "normal_deviation.hpp"
#include "util/parallel.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/scene/world_container.hpp"
#include <random>
#include <stdexcept>

namespace mufflon::renderer {

using namespace importance;

namespace {

float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CpuImportanceDecimater::CpuImportanceDecimater() {
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuImportanceDecimater::on_scene_load() {
	m_importanceMap.clear();
}

void CpuImportanceDecimater::post_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());

	if(m_sceneDesc.numInstances != m_sceneDesc.numLods)
		throw std::runtime_error("We do not support instancing yet");

	// Initialize the importance map
	// TODO: how to deal with instancing
	if(!m_params.keepImportance || !m_gotImportance) {
		initialize_importance_map();
		m_gotImportance = false;
	}
}

void CpuImportanceDecimater::iterate() {
	if(m_currentImportanceIteration >= m_params.importanceIterations) {
		if(!m_finishedDecimation) {
			if(m_params.decimationEnabled)
				this->decimate();
			m_finishedDecimation = true;
			m_gotImportance = true;
			m_reset = true;
		} else {
			const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
			for(int i = 0; i < (int)NUM_PIXELS; ++i) {
				this->pt_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
			}
		}
	} else {
		if(!m_params.keepImportance || !m_gotImportance) {
			// TODO: enable decimation
			logInfo("Importance iteration (", m_currentImportanceIteration + 1, " of ", m_params.importanceIterations, ")");
			gather_importance();
			++m_currentImportanceIteration;
		}

		if(m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			compute_max_importance();
			logInfo("Max. importance: ", m_maxImportance);
			display_importance();
		}
	}
}


void CpuImportanceDecimater::gather_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < (int)NUM_PIXELS; ++i) {
		this->importance_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
	}
	// TODO: allow for this with proper reset "events"
	m_importanceMap.update_normalized();
}

void CpuImportanceDecimater::decimate() {
	logInfo("Finished importance gathering; starting decimation (target reduction of ", m_params.reduction, ")");
	u32 vertexOffset = 0u;
	u32 meshIndex = 0u;
	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				auto& lod = object.first->get_lod(i);
				auto& polygons = lod.get_geometry<scene::geometry::Polygons>();
				const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());

				if(static_cast<int>(vertexCount) < m_params.threshold) {
					logInfo("Skipping object ", object.first->get_object_id(), ", LoD ", i,
							" (too little vertices)");
				} else {
					const u32 targetVertexCount = static_cast<u32>(vertexCount * (1.f - m_params.reduction));
					logInfo("Decimating object ", object.first->get_object_id(), ", LoD ", i,
							" (", vertexCount, " => ", targetVertexCount, ")");

					auto& mesh = polygons.get_mesh();
					OpenMesh::VPropHandleT<float> impHdl;

					// Create decimater and attach modules
					auto decimater = polygons.create_decimater();

					ModImportance<>::Handle impModHdl;
					decimater.add(impModHdl);
					decimater.module(impModHdl).set_importance_map(m_importanceMap, meshIndex);
					NormalDeviationModule<>::Handle normModHdl;
					decimater.add(normModHdl);
					decimater.module(normModHdl).set_max_deviation(Degrees(60.0));
					polygons.decimate(decimater, targetVertexCount, true);

					lod.clear_accel_structure();
				}
				vertexOffset += vertexCount;
				++meshIndex;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_reset = true;
}

void CpuImportanceDecimater::importance_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	// We gotta keep track of our vertices
	thread_local std::vector<ImpPathVertex> vertices(std::max(2, m_params.maxPathLength + 1));
	// Create a start for the path
	(void)ImpPathVertex::create_camera(&vertices.front(), &vertices.front(), m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

	float sharpness = 1.f;
	
	// Andreas' algorithm mapped to path tracing:
	// Increasing importance for photons is equivalent to increasing
	// importance by the irradiance. Thus we need to compute "both
	// direct and indirect" irradiance for the path tracer as well.
	// They differ, however, in the types of paths that they
	// preferably find.

	int pathLen = 0;
	do {
		vertices[pathLen].ext().pathRadiance = ei::Vec3{ 0.f };
		// Add direct contribution as importance as well
		if(pathLen > 0 && pathLen + 1 <= m_params.maxPathLength) {
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertices[pathLen].get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			Pixel projCoord;
			auto value = vertices[pathLen].evaluate(nee.direction, m_sceneDesc.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection(
					m_sceneDesc, { vertices[pathLen].get_position() , nee.direction },
					vertices[pathLen].get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));

					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;
					// Add the importance
					const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
					const float weightedIrradianceLuminance = mis * get_luminance(throughput.weight * irradiance) * (1.f - ei::abs(vertices[pathLen - 1].ext().outCos));

					const auto& hitId = vertices[pathLen].get_primitive_id();
					const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
					const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
					const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
					for(u32 i = 0u; i < numVertices; ++i) {
						const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
						m_importanceMap.add(hitId.instanceId, vertexIndex, weightedIrradianceLuminance);
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;

		if(!walk(m_sceneDesc, vertices[pathLen], rnd, -1.0f, false, throughput, vertices[pathLen + 1], sample))
			break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().updateBxdf(sample, throughput);

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;

		if(pathLen > 0) {
			// Direct hits are being scaled down in importance by a sigmoid of the BxDF to get an idea of the "sharpness"
			const float importance = sharpness * (1.f - ei::abs(ei::dot(vertices[pathLen].get_normal(), vertices[pathLen].get_incident_direction())));
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
				m_importanceMap.add(hitId.instanceId, vertexIndex, importance);
			}

			const ei::Vec3 bxdf = vertices[pathLen].ext().bxdfPdf * (float)vertices[pathLen].ext().pdf;
			sharpness *= 2.f / (1.f + ei::exp(-get_luminance(bxdf))) - 1.f;

		}

		++pathLen;
	} while(pathLen < m_params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{0.f};
	float accumThroughout = 1.f;
	for(int p = pathLen - 2; p >= 1; --p) {
		accumRadiance = vertices[p].ext().throughput * accumRadiance + vertices[p + 1].ext().pathRadiance;
		const ei::Vec3 irradiance = vertices[p].ext().accumThroughput * vertices[p].ext().outCos * accumRadiance;

		const auto& hitId = vertices[p].get_primitive_id();
		const auto& lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitId.primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

		const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
		for(u32 i = 0u; i < numVertices; ++i) {
			const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
			m_importanceMap.add(hitId.instanceId, vertexIndex, importance);
		}
	}
}

void CpuImportanceDecimater::pt_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	// Create a start for the path
	(void)PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());


	int pathLen = 0;
	do {
		if(pathLen > 0.0f && pathLen + 1 <= m_params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertex.get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			Pixel projCoord;
			auto value = vertex.evaluate(nee.direction, m_sceneDesc.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection(
					m_sceneDesc, { vertex.get_position() , nee.direction },
					vertex.get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					m_outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
											value.cosOut, radiance * mis);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		VertexSample sample;
		if(!walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex, sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / sample.pdfF);
					background.value *= mis;
					m_outputBuffer.contribute(coord, throughput, background.value,
											  ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											  ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}

		if(pathLen == 0 && m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ this->query_importance(vertex.get_position(), vertex.get_primitive_id()) });
		}

		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= m_params.maxPathLength) {
			Spectrum emission = vertex.get_emission().value;
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(m_sceneDesc.lightTree, vertex.get_primitive_id(),
												  vertex.get_surface_params(),
												  lastPosition, scene::lights::guide_flux);
				float mis = pathLen == 1 ? 1.0f
					: 1.0f / (1.0f + backwardPdf / vertex.ext().incidentPdf);
				emission *= mis;
			}
			m_outputBuffer.contribute(coord, throughput, emission, vertex.get_position(),
									vertex.get_normal(), vertex.get_albedo());
		}
	} while(pathLen < m_params.maxPathLength);
}

void CpuImportanceDecimater::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		for(u32 v = 0u; v < m_importanceMap.get_vertex_count(i); ++v) {
			m_maxImportance = std::max(m_maxImportance, m_importanceMap.normalized(i, v));
		}
	}
}

void CpuImportanceDecimater::display_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const ei::IVec2 coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		//m_params.maxPathLength = 2;

		math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
		PtPathVertex vertex;
		// Create a start for the path
		(void)PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		VertexSample sample;
		if(walk(m_sceneDesc, vertex, rnd, -1.0f, false, throughput, vertex, sample))
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ this->query_importance(vertex.get_position(), vertex.get_primitive_id()) });
	}

}

float CpuImportanceDecimater::query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId) {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];

	float importance = 0.f;

	// Compute the closest vertex
	const u32 vertexCount = ((u32)hitId.primId < lod.polygon.numTriangles) ? 3u : 4u;
	if(vertexCount == 3u) {
		// Triangle
		const u32 i0 = lod.polygon.vertexIndices[3u * hitId.primId + 0u];
		const u32 i1 = lod.polygon.vertexIndices[3u * hitId.primId + 1u];
		const u32 i2 = lod.polygon.vertexIndices[3u * hitId.primId + 2u];
		const float d0 = ei::lensq(hitPoint - lod.polygon.vertices[i0]);
		const float d1 = ei::lensq(hitPoint - lod.polygon.vertices[i1]);
		const float d2 = ei::lensq(hitPoint - lod.polygon.vertices[i2]);
		if(d0 < d1) {
			if(d0 < d2)
				importance = m_importanceMap.normalized(hitId.instanceId, i0);
			else
				importance = m_importanceMap.normalized(hitId.instanceId, i2);
		} else {
			if(d1 < d2)
				importance = m_importanceMap.normalized(hitId.instanceId, i1);
			else
				importance = m_importanceMap.normalized(hitId.instanceId, i2);
		}

	} else {
		mAssert(vertexCount == 4u);
		// Quad
		const u32 vertexOffset = 3u * static_cast<u32>(lod.polygon.numTriangles);
		const u32 i0 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 0u];
		const u32 i1 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 1u];
		const u32 i2 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 2u];
		const u32 i3 = lod.polygon.vertexIndices[vertexOffset + 4u * hitId.primId + 3u];
		const float d0 = ei::lensq(hitPoint - lod.polygon.vertices[i0]);
		const float d1 = ei::lensq(hitPoint - lod.polygon.vertices[i1]);
		const float d2 = ei::lensq(hitPoint - lod.polygon.vertices[i2]);
		const float d3 = ei::lensq(hitPoint - lod.polygon.vertices[i3]);

		if(d0 < d1) {
			if(d0 < d2) {
				if(d0 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i0);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			} else {
				if(d2 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i2);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			}
		} else {
			if(d1 < d2) {
				if(d1 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i1);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			} else {
				if(d2 < d3)
					importance = m_importanceMap.normalized(hitId.instanceId, i2);
				else
					importance = m_importanceMap.normalized(hitId.instanceId, i3);
			}
		}
	}
	return importance / m_maxImportance;
}

void CpuImportanceDecimater::initialize_importance_map() {
	std::vector<scene::geometry::PolygonMeshType*> meshes;

	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				meshes.push_back(&object.first->get_lod(i).template get_geometry<scene::geometry::Polygons>().get_mesh());
			}
		}
	}
	// We do it in this ugly fashion because create -> move -> destroy-old doesn't work due to how OpenMesh treats properties
	m_importanceMap.~ImportanceMap();
	new(&m_importanceMap) ImportanceMap(std::move(meshes));
}

void CpuImportanceDecimater::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer