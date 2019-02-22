#include "cpu_importance.hpp"
#include "decimater.hpp"
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

inline void atomic_add(std::atomic<float>& af, const float diff) {
	float expected = af.load();
	float desired;
	do {
		desired = expected + diff;
	} while(!af.compare_exchange_weak(expected, desired));
}

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

CpuImportanceDecimater::CpuImportanceDecimater() {
	// TODO: init one RNG per thread?
	std::random_device rndDev;
	m_rngs.emplace_back(static_cast<u32>(rndDev()));
}

void CpuImportanceDecimater::on_descriptor_requery() {
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
	// TODO: enable decimation
	logInfo("Importance iteration (", m_currentImportanceIteration + 1, " of ", m_params.importanceIterations, ")");
	gather_importance();
	++m_currentImportanceIteration;

	// TODO: visualize importance
	if(m_params.importanceIterations > 0 && m_outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
		compute_max_importance();
		logWarning("Max importance: ", m_maxImportance);
		display_importance();
	}
}


void CpuImportanceDecimater::gather_importance() {
	const u32 NUM_PIXELS = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < (int)NUM_PIXELS; ++i) {
		this->importance_sample(Pixel{ i % m_outputBuffer.get_width(), i / m_outputBuffer.get_width() });
	}
	// TODO: allow for this with proper reset "events"
}

void CpuImportanceDecimater::decimate() {
	constexpr StringView importanceAttrName{ "importance" };

	logInfo("Finished importance gathering; starting decimation (target reduction of ", m_params.reduction, ")");
	u32 vertexOffset = 0u;
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
					mesh.add_property(impHdl, std::string(importanceAttrName));

					for(u32 v = 0u; v < vertexCount; ++v)
						mesh.property(impHdl, OpenMesh::VertexHandle {(int)v}) = m_importanceMap[vertexOffset + v].load();

					// Create decimater and attach modules
					/*auto decimater = polygons.create_decimater();

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
					polygons.decimate(decimater, targetVertexCount);*/

					lod.clear_accel_structure();
				}
				vertexOffset += vertexCount;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, m_outputBuffer.get_resolution());
}

void CpuImportanceDecimater::importance_sample(const Pixel coord) {
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
					const float weightedIrradianceLuminance = mis * get_luminance(throughput.weight * irradiance);
					mAssert(lod != nullptr);
					for(u32 i = 0u; i < numVertices; ++i) {
						const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[pathLen - 1].primId + i];
						const u32 index = m_vertexOffsets[hitIds[pathLen - 1].instanceId] + vertexIndex;
						atomic_add(m_importanceMap[index], 1.f);
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
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[pathLen].primId + i];
				const u32 index = m_vertexOffsets[hitIds[pathLen].instanceId] + vertexIndex;

				atomic_add(m_importanceMap[index], sharpness);
			}

			sharpness *= 2.f / (1.f + ei::exp(-get_luminance(bxdf))) - 1.f;
			if(isnan(sharpness))
				__debugbreak();

		}

		++pathLen;
		pathRadiance[pathLen] = ei::Vec3{ 0.f };
	} while(pathLen < m_params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{0.f};
	float accumThroughout = 1.f;
	for(int p = pathLen - 1; p >= 1; --p) {
		accumRadiance = vertexThroughput[p] * accumRadiance + pathRadiance[p + 1];
		const ei::Vec3 irradiance = vertexAccumThroughput[p] * outCos[p] * accumRadiance;

		const auto& lod = &m_sceneDesc.lods[m_sceneDesc.lodIndices[hitIds[p].instanceId]];
		const u32 numVertices = hitIds[p].primId < (i32)lod->polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = hitIds[p].primId < (i32)lod->polygon.numTriangles ? 0u : 3u * lod->polygon.numTriangles;
		for(u32 i = 0u; i < numVertices; ++i) {
			const u32 vertexIndex = lod->polygon.vertexIndices[vertexOffset + numVertices * hitIds[p].primId + i];
			const u32 index = m_vertexOffsets[hitIds[p].instanceId] + vertexIndex;

			atomic_add(m_importanceMap[index], get_luminance(irradiance));
		}
	}
}

void CpuImportanceDecimater::pt_sample(const Pixel coord) {
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
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ compute_importance(vertex->get_primitive_id()) });
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

void CpuImportanceDecimater::compute_max_importance() {
	m_maxImportance = 0.f;

	// Compute the maximum normalized importance for visualization
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[i]];
		// First triangles, then quads
		for(u32 t = 0u; t < lod.polygon.numTriangles; ++t) {
			const u32 A = lod.polygon.vertexIndices[3u * t + 0u];
			const u32 B = lod.polygon.vertexIndices[3u * t + 1u];
			const u32 C = lod.polygon.vertexIndices[3u * t + 2u];
			const float area = ei::surface(ei::Triangle{
				lod.polygon.vertices[A], lod.polygon.vertices[B],
				lod.polygon.vertices[C] });
			const float importance = (m_importanceMap[m_vertexOffsets[i] + A]
									  + m_importanceMap[m_vertexOffsets[i] + B]
									  + m_importanceMap[m_vertexOffsets[i] + C]) / 3.f;

			m_maxImportance = std::max(importance / area, m_maxImportance);
		}

		for(u32 q = 0u; q < lod.polygon.numQuads; ++q) {
			const u32 A = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * q + 0u];
			const u32 B = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * q + 1u];
			const u32 C = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * q + 2u];
			const u32 D = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * q + 3u];
			const float area = ei::surface(ei::Triangle{
				lod.polygon.vertices[A], lod.polygon.vertices[B],
				lod.polygon.vertices[C] }) + ei::surface(ei::Triangle{
				lod.polygon.vertices[A], lod.polygon.vertices[C],
				lod.polygon.vertices[D] });
			const float importance = (m_importanceMap[m_vertexOffsets[i] + A]
									  + m_importanceMap[m_vertexOffsets[i] + B]
									  + m_importanceMap[m_vertexOffsets[i] + C]
									  + m_importanceMap[m_vertexOffsets[i] + D]) / 4.f;

			m_maxImportance = std::max(importance / area, m_maxImportance);
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
			m_outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ compute_importance(vertex->get_primitive_id()) });
	}

}

float CpuImportanceDecimater::compute_importance(const scene::PrimitiveHandle& hitId) {
	const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
	const u32 vertexCount = ((u32)hitId.primId < lod.polygon.numTriangles) ? 3u : 4u;
	const u32 vertexOffset = m_vertexOffsets[hitId.instanceId];

	// Normalize the importance by the primitive's area
	float importance;
	if(vertexCount == 3u) {
		const u32 A = lod.polygon.vertexIndices[3u * hitId.primId + 0u];
		const u32 B = lod.polygon.vertexIndices[3u * hitId.primId + 1u];
		const u32 C = lod.polygon.vertexIndices[3u * hitId.primId + 2u];
		const float area = ei::surface(ei::Triangle{
			lod.polygon.vertices[A], lod.polygon.vertices[B],
			lod.polygon.vertices[C] });
		importance = (m_importanceMap[vertexOffset + A]
					  + m_importanceMap[vertexOffset + B]
					  + m_importanceMap[vertexOffset + C]) / (area * 3.f);
	} else {
		const u32 quadId = hitId.primId - lod.polygon.numTriangles;
		const u32 A = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * quadId + 0u];
		const u32 B = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * quadId + 1u];
		const u32 C = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * quadId + 2u];
		const u32 D = lod.polygon.vertexIndices[3u * lod.polygon.numTriangles + 4u * quadId + 3u];
		const float area = ei::surface(ei::Triangle{
			lod.polygon.vertices[A], lod.polygon.vertices[B],
			lod.polygon.vertices[C] }) + ei::surface(ei::Triangle{
			lod.polygon.vertices[A], lod.polygon.vertices[C],
			lod.polygon.vertices[D] });
		importance = (m_importanceMap[vertexOffset + A]
					  + m_importanceMap[vertexOffset + B]
					  + m_importanceMap[vertexOffset + C]
					  + m_importanceMap[vertexOffset + D]) / (area * 3.f);
	}

	return importance / m_maxImportance;
}

void CpuImportanceDecimater::initialize_importance_map() {
	m_vertexCount = 0u;
	m_vertexOffsets = make_udevptr_array<Device::CPU, u32>(m_sceneDesc.numInstances);
	for(i32 i = 0u; i < m_sceneDesc.numInstances; ++i) {
		m_vertexOffsets[i] = m_vertexCount;
		m_vertexCount += m_sceneDesc.lods[m_sceneDesc.lodIndices[i]].polygon.numVertices;
	}
	// Allocate sufficient buffer
	m_importanceMap = make_udevptr_array<Device::CPU, std::atomic<float>>(m_vertexCount);
	// Initilaize importance with zero
	for(u32 i = 0u; i < m_vertexCount; ++i)
		m_importanceMap[i].store(0.f);
}

void CpuImportanceDecimater::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer