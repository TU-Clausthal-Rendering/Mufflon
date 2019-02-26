#pragma once

#include "cpu_silhouette.hpp"
#include "sil_decimater.hpp"
#include "util/parallel.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/scene/world_container.hpp"
#include <random>
#include <stdexcept>

namespace mufflon::renderer {

namespace {

struct LightData {
	scene::lights::LightType type;
	float flux;
	u32 offset;
};

inline LightData get_light_data(const u32 lightIdx, const scene::lights::LightSubTree& tree) {
	// Special case for only single light
	if(tree.lightCount == 1) {
		return LightData{
			static_cast<scene::lights::LightType>(tree.root.type),
			tree.root.flux,
			0u
		};
	}

	// Determine what level of the tree the light is on
	const u32 level = std::numeric_limits<u32>::digits - 1u - static_cast<u32>(std::log2(tree.internalNodeCount + lightIdx + 1u));
	// Determine the light's node index within its level
	const u32 levelIndex = (tree.internalNodeCount + lightIdx) - ((1u << level) - 1u);
	// The corresponding parent node's level index is then the level index / 2
	const u32 parentLevelIndex = levelIndex / 2u;
	// Finally, compute the tree index of the node
	const u32 parentIndex = (1u << (level - 1u)) - 1u + parentLevelIndex;
	const scene::lights::LightSubTree::Node& node = *tree.get_node(parentIndex * sizeof(scene::lights::LightSubTree::Node));

	// Left vs. right node
	// TODO: for better divergence let all even indices be processes first, then the uneven ones
	if(levelIndex % 2 == 0) {
		mAssert(node.left.type < static_cast<u16>(scene::lights::LightType::NUM_LIGHTS));
		return LightData{
			static_cast<scene::lights::LightType>(node.left.type),
			node.left.flux, node.left.offset
		};
	} else {
		mAssert(node.right.type < static_cast<u16>(scene::lights::LightType::NUM_LIGHTS));
		return LightData{
			static_cast<scene::lights::LightType>(node.right.type),
			node.right.flux, node.right.offset
		};
	}
}

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

inline void atomic_add(std::atomic<float>& af, const float diff) {
	float expected = af.load();
	float desired;
	do {
		desired = expected + diff;
	} while(!af.compare_exchange_weak(expected, desired));
}

} // namespace

CpuShadowSilhouettes::CpuShadowSilhouettes()
{
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

bool CpuShadowSilhouettes::pre_iteration(OutputHandler& outputBuffer) {
	if(!m_reset && (int)m_currentDecimationIteration < m_params.decimationIterations) {
		if(m_params.decimationEnabled)
			this->decimate();
		m_finishedDecimation = true;
		m_reset = true;
	}
	return RendererBase<Device::CPU>::pre_iteration(outputBuffer);
}

void CpuShadowSilhouettes::on_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());

	if(m_sceneDesc.numInstances != m_sceneDesc.numLods)
		throw std::runtime_error("We do not support instancing yet");

	// Initialize the importance map
	// TODO: how to deal with instancing
	initialize_importance_map();
	m_currentDecimationIteration = 0u;
}

void CpuShadowSilhouettes::iterate() {
	// TODO: incorporate decimation/undecimation loop!

	if((int)m_currentDecimationIteration < m_params.decimationIterations) {
		gather_importance();

		if(m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			compute_max_importance();
			logInfo("Max. importance: ", m_maxImportance);
			display_importance();
		}
	} else {
		const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
		for(int i = 0; i < (int)NUM_PIXELS; ++i) {
			this->pt_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
		}
	}
}


void CpuShadowSilhouettes::gather_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < m_params.importanceIterations * (int)NUM_PIXELS; ++i) {
		const int pixel = i / m_params.importanceIterations;
		this->importance_sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() });
	}
	// TODO: allow for this with proper reset "events"
	m_importanceMap.update_normalized();
}

void CpuShadowSilhouettes::decimate() {
	constexpr StringView importanceAttrName{ "importance" };


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


					ImportanceModule<>::Handle impModHdl;
					decimater.add(impModHdl);
					decimater.module(impModHdl).set_importance_map(m_importanceMap, meshIndex);
					/*MaxNormalDeviation<>::Handle normModHdl;
					decimater.add(normModHdl);
					decimater.module(normModHdl).set_max_deviation(60.0);*/
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

void CpuShadowSilhouettes::undecimate() {
	/*constexpr StringView importanceAttrName{ "importance" };

	u32 vertexOffset = 0u;
	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				auto& lod = object.first->get_lod(i);
				auto& polygons = lod.get_geometry<scene::geometry::Polygons>();
				const u32 vertexCount = static_cast<u32>(polygons.get_vertex_count());

				auto& mesh = polygons.get_mesh();

				for(auto edge : mesh.edges()) {
					auto heh = mesh.halfedge_handle(edge, 0);
					auto v0 = mesh.from_vertex_handle(heh);
					auto v1 = mesh.to_vertex_handle(heh);
					auto imp0 = m_importanceMap[vertexOffset + v0.idx()].load();
					auto imp1 = m_importanceMap[vertexOffset + v1.idx()].load();

					if(imp0 > 0.f || imp1 > 0.f) {
						if(imp0 > imp1) {

						} else {

						}
					}
				}

				if(static_cast<int>(vertexCount) < m_params.threshold) {
					logInfo("Skipping object ", object.first->get_object_id(), ", LoD ", i,
							" (too little vertices)");
				} else {
					const u32 targetVertexCount = static_cast<u32>(vertexCount * (1.f - m_params.reduction));
					logInfo("Decimating object ", object.first->get_object_id(), ", LoD ", i,
							" (", vertexCount, " => ", targetVertexCount, ")");

					auto& mesh = polygons.get_mesh();
					OpenMesh::VPropHandleT<float> impHdl;
					mesh.add_property(impHdl, std::string(importanceAttrName));

					for(u32 v = 0u; v < vertexCount; ++v)
						mesh.property(impHdl, OpenMesh::VertexHandle {(int)v}) = m_importanceMap[vertexOffset + v].load();

					// Create decimater and attach modules
					auto decimater = polygons.create_decimater();

					ImportanceModule<scene::geometry::PolygonMeshType>::Handle impModHdl;
					ImportanceBinaryModule<scene::geometry::PolygonMeshType>::Handle impBinModHdl;
					decimater.add(impModHdl);
					decimater.add(impBinModHdl);
					decimater.module(impModHdl).set_importance_property(impHdl);
					decimater.module(impBinModHdl).set_importance_property(impHdl);
					//OpenMesh::Decimater::ModNormalFlippingT<scene::geometry::PolygonMeshType>::Handle normal_flipping;
					//decimater.add(normal_flipping);
					//decimater.module(normal_flipping).set_max_normal_deviation(m_params.maxNormalDeviation);

					// Perform actual decimation
					// TODO: only create LoD
					polygons.decimate(decimater, targetVertexCount, false);

					lod.clear_accel_structure();
				}
				vertexOffset += vertexCount;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, m_outputBuffer.get_resolution());*/
}

void CpuShadowSilhouettes::importance_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");

	// TODO: depends on path length
	scene::PrimitiveHandle hitIds[16];
	float outCos[16];					// Outgoing cosine
	ei::Vec3 vertexThroughput[16];
	ei::Vec3 vertexAccumThroughput[16];
	ei::Vec3 pathRadiance[16];
	float sharpness = 1.f;

	ei::Vec3 bxdfPdf[16];				// BxDF/PDF in path direction of eye


	if(m_params.maxPathLength > 16)
		throw std::runtime_error("Max path length too high");

	const scene::LodDescriptor<Device::CPU>* lod = nullptr;
	u32 numVertices = 0;
	u32 vertexOffset = 0;


	// Andreas' algorithm mapped to path tracing:
	// Increasing importance for photons is equivalent to increasing
	// importance by the irradiance. Thus we need to compute "both
	// direct and indirect" irradiance for the path tracer as well.
	// They differ, however, in the types of paths that they
	// preferably find.

	int pathLen = 0;
	do {
		pathRadiance[pathLen] = ei::Vec3{ 0.f };
		// Add direct contribution as importance as well
		if(pathLen > 0 && pathLen + 1 <= m_params.maxPathLength) {
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
							   vertex->get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex->evaluate(nee.direction, m_sceneDesc.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
					m_sceneDesc, { vertex->get_position() , nee.direction },
					vertex->get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));

					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					pathRadiance[pathLen] = vertexThroughput[pathLen - 1] * mis * radiance * value.cosOut;
					// Add the importance
					const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
					const float weightedIrradianceLuminance = mis * get_luminance(throughput.weight * irradiance) * (1.f - ei::abs(outCos[pathLen - 1]));
					mAssert(lod != nullptr);
					for(u32 i = 0u; i < numVertices; ++i) {
						const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[pathLen - 1].primId + i];
						m_importanceMap.add(hitIds[pathLen - 1].instanceId, vertexIndex, weightedIrradianceLuminance);
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;

		if(!walk(m_sceneDesc, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir, &vertexThroughput[pathLen]))
			break;

		hitIds[pathLen] = vertex->get_primitive_id();
		// Compute BRDF
		vertexAccumThroughput[pathLen] = throughput.weight;
		outCos[pathLen] = -ei::dot(vertex->get_normal(), lastDir.direction);
		bxdfPdf[pathLen] = vertexThroughput[pathLen] / outCos[pathLen];
		const ei::Vec3 bxdf = bxdfPdf[pathLen] * (float)lastDir.pdf;

		lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitIds[pathLen].instanceId]];
		numVertices = hitIds[pathLen].primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		vertexOffset = hitIds[pathLen].primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

		if(pathLen > 0) {
			// Direct hits are being scaled down in importance by a sigmoid of the BxDF to get an idea of the "sharpness"
			const float importance = sharpness * (1.f - ei::abs(ei::dot(vertex->get_normal(), vertex->get_incident_direction())));
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[pathLen].primId + i];
				m_importanceMap.add(hitIds[pathLen].instanceId, vertexIndex, importance);
			}

			sharpness *= 2.f / (1.f + ei::exp(-get_luminance(bxdf))) - 1.f;

		}

		++pathLen;
	} while(pathLen < m_params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{ 0.f };
	float accumThroughout = 1.f;
	for(int p = pathLen - 2; p >= 1; --p) {
		accumRadiance = vertexThroughput[p] * accumRadiance + pathRadiance[p + 1];
		const ei::Vec3 irradiance = vertexAccumThroughput[p] * outCos[p] * accumRadiance;

		const auto& lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitIds[p].instanceId]];
		const u32 numVertices = hitIds[p].primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitIds[p].primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;

		const float importance = get_luminance(irradiance) * (1.f - ei::abs(outCos[p]));
		for(u32 i = 0u; i < numVertices; ++i) {
			const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[p].primId + i];
			m_importanceMap.add(hitIds[p].instanceId, vertexIndex, importance);
		}
	}
}

void CpuShadowSilhouettes::pt_sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");


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
							   vertex->get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex->evaluate(nee.direction, m_sceneDesc.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
					m_sceneDesc, { vertex->get_position() , nee.direction },
					vertex->get_primitive_id(), nee.dist);
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
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		if(!walk(m_sceneDesc, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, lastDir.direction);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
					background.value *= mis;
					m_outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}

		if(pathLen == 0 && m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex->get_position(), vertex->get_primitive_id()) });
		}

		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= m_params.maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(m_sceneDesc.lightTree, vertex->get_primitive_id(),
												  vertex->get_surface_params(),
												  lastPosition, scene::lights::guide_flux);
				float mis = pathLen == 1 ? 1.0f
					: 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				emission *= mis;
			}
			m_outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
									  vertex->get_normal(), vertex->get_albedo());
		}
	} while(pathLen < m_params.maxPathLength);
}

void CpuShadowSilhouettes::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
//#pragma omp parallel for reduction(max:m_maxImportance)
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		for(u32 v = 0u; v < m_importanceMap.get_vertex_count(i); ++v) {
			m_maxImportance = std::max(m_maxImportance, m_importanceMap.normalized(i, v));
		}
	}
}

void CpuShadowSilhouettes::display_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const ei::IVec2 coord{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		//m_params.maxPathLength = 2;

		math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
		u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
		PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
		// Create a start for the path
		int s = PtPathVertex::create_camera(vertex, vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());
		mAssertMsg(s < 256, "vertexBuffer overflow.");


		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		if(walk(m_sceneDesc, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir))
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ query_importance(vertex->get_position(), vertex->get_primitive_id()) });
	}

}



float CpuShadowSilhouettes::query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId) {
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

bool CpuShadowSilhouettes::trace_shadow_silhouette(const ei::Ray& shadowRay, const PtPathVertex& vertex,
												   const scene::PrimitiveHandle& firstHit,
												   const float firstHitT, const float lightDist, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	const ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * firstHitT, shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, backfaceRay, firstHit,
																			  lightDist - firstHitT + DIST_EPSILON);
	// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
	if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex.get_primitive_id()
	   && secondHit.hitId.instanceId == firstHit.instanceId) {
		// Check for silhouette - get the vertex indices of the primitives
		const auto& obj = m_sceneDesc.lods[m_sceneDesc.lodIndices[firstHit.instanceId]];
		const i32 firstNumVertices = firstHit.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 firstPrimIndex = firstHit.primId - (firstHit.primId < (i32)obj.polygon.numTriangles
													  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
															  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 firstVertOffset = firstHit.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
		const i32 secondVertOffset = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;

		// Check if we have "shared" vertices: cannot do it by index, since they might be
		// split vertices, but instead need to go by proximity
		i32 sharedVertices = 0;
		i32 edgeIdxFirst[2];
		i32 edgeIdxSecond[2];
		for(i32 i0 = 0; i0 < firstNumVertices; ++i0) {
			for(i32 i1 = 0; i1 < secondNumVertices; ++i1) {
				const i32 idx0 = obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + i0];
				const i32 idx1 = obj.polygon.vertexIndices[secondVertOffset + secondNumVertices * secondPrimIndex + i1];
				const ei::Vec3& p0 = obj.polygon.vertices[idx0];
				const ei::Vec3& p1 = obj.polygon.vertices[idx1];
				if(idx0 == idx1 || p0 == p1) {
					edgeIdxFirst[sharedVertices] = idx0;
					edgeIdxSecond[sharedVertices] = idx1;
					++sharedVertices;
				}
				if(sharedVertices >= 2)
					break;
			}
		}

		if(sharedVertices >= 1) {
			// Got at least a silhouette point - now make sure we're seeing the silhouette
			const ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (firstHitT + secondHit.hitT), shadowRay.direction };

			const auto thirdHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, silhouetteRay, secondHit.hitId,
																					 lightDist - firstHitT - secondHit.hitT + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				for(i32 i = 0; i < sharedVertices; ++i) {
					// x86_64 doesn't support atomic_fetch_add for floats FeelsBadMan
					m_importanceMap.add(firstHit.instanceId, edgeIdxFirst[i], importance);
					m_importanceMap.add(firstHit.instanceId, edgeIdxSecond[i], importance);
				}
				return true;
			} else {
				mAssert(thirdHit.hitId.instanceId >= 0);
				// TODO: store a shadow photon?
			}
		}
	}
	return false;
}

void CpuShadowSilhouettes::initialize_importance_map() {
	std::vector<scene::geometry::PolygonMeshType*> meshes;

	for(auto object : m_currentScene->get_objects()) {
		for(u32 i = 0u; i < object.first->get_lod_slot_count(); ++i) {
			if(object.first->has_lod_available(i)) {
				meshes.push_back(&object.first->get_lod(i).template get_geometry<scene::geometry::Polygons>().get_mesh());
			}
		}
	}
	m_importanceMap = ImportanceMap(std::move(meshes));
}

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer