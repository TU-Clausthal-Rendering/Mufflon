#pragma once

#include "cpu_silhouette.hpp"
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

using namespace silhouette;

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

void CpuShadowSilhouettes::iterate(OutputHandler& outputBuffer) {
	// (Re) create the random number generators
	if(m_rngs.size() != outputBuffer.get_num_pixels()
	   || m_reset)
		init_rngs(outputBuffer.get_num_pixels());

	RenderBuffer<Device::CPU> buffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	if(m_reset) {
		// Reacquire scene descriptor
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, outputBuffer.get_resolution());

		if(m_sceneDesc.numInstances != m_sceneDesc.numLods)
			throw std::runtime_error("We do not support instancing yet");

		// Initialize the importance map
		// TODO: how to deal with instancing
		if(!m_params.keepImportance || !m_gotImportance) {
			initialize_importance_map();
			m_currentDecimationIteration = 0u;
			m_gotImportance = false;
		}
	}

	if(m_currentDecimationIteration < (u32)m_params.decimationIterations && (!m_params.keepImportance || !m_gotImportance)) {
		gather_importance(buffer);
		++m_currentDecimationIteration;
		logInfo("Finished importance gathering (iteration ", m_currentDecimationIteration, " of ", m_params.decimationIterations, ")");

		// TODO: visualize importance
		if(m_params.importanceIterations > 0 && outputBuffer.get_target().is_set(1 << RenderTargets::IMPORTANCE)) {
			compute_max_importance();
			display_importance(buffer);
		}

		// TODO: decimate
		//decimate(buffer.get_resolution());
	} else {
		m_gotImportance = true;

		// Make sure we compute the maximum importance for visualization at least once
		if((m_reset || !m_finishedDecimation) && m_params.importanceIterations > 0 && outputBuffer.get_target().is_set(1 << RenderTargets::IMPORTANCE))
			compute_max_importance();

		if(!m_finishedDecimation) {
			buffer = outputBuffer.begin_iteration<Device::CPU>(true);
			// TODO: finalize decimation
			m_finishedDecimation = true;
			logInfo("Finished decimating, beginning normal rendering...");
		}

		logWarning("Maximum importance: ", m_maxImportance);

		// TODO: call sample in a parallel way for each output pixel
		// TODO: better pixel order?
		// TODO: different scheduling?
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
			this->pt_sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer, m_sceneDesc);
		}
	}

	m_reset = false;

	outputBuffer.end_iteration<Device::CPU>();
}


void CpuShadowSilhouettes::gather_importance(RenderBuffer<Device::CPU>& buffer) {
	const u32 NUM_PIXELS = buffer.get_height() * buffer.get_width();
	const int importanceIterations = m_params.importanceIterations * NUM_PIXELS;
	logInfo("Starting importance gathering (", m_params.importanceIterations, " iterations)");
#pragma PARALLEL_FOR
	for(int i = 0; i < m_params.importanceIterations * (int)NUM_PIXELS; ++i) {
		const int pixel = i / m_params.importanceIterations;
		this->importance_sample(Pixel{ pixel % buffer.get_width(), pixel / buffer.get_width() },
								buffer, m_sceneDesc);
	}
	// TODO: allow for this with proper reset "events"
}

void CpuShadowSilhouettes::decimate(const ei::IVec2& resolution) {
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
					polygons.decimate(decimater, targetVertexCount);

					lod.clear_accel_structure();
				}
				vertexOffset += vertexCount;
			}
		}
	}

	// We need to re-build the scene
	m_currentScene->clear_accel_structure();
	m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, resolution);
}

void CpuShadowSilhouettes::importance_sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
											 const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");


	int pathLen = 0;
	do {
		// TODO: factor lighting into importance

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		if(!walk(scene, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir))
			break;

		// View ray importance
		if(pathLen == 0 || m_params.enableIndirectImportance) {
			// TODO: factor in BRDF
			const auto& hitId = vertex->get_primitive_id();
			const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];

			const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
			const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
				// TODO: proper amount of importance!
				const u32 index = m_vertexOffsets[hitId.instanceId] + vertexIndex;
				atomic_add(m_importanceMap[index], 1.f);
			}
		}

		// First hit: factor in shadow silhouettes
		// TODO: base this on contribution instead of path length
		if(pathLen == 0) {
			// Check silhouettes for all lights
			// TODO: improve upon this?
			for(u32 lightIdx = 0u; lightIdx < scene.lightTree.posLights.lightCount; ++lightIdx) {
				auto lightData = get_light_data(lightIdx, scene.lightTree.posLights);
				const char* lightMem = scene.lightTree.posLights.memory + lightData.offset;
				const ei::Vec3 lightCenter = scene::lights::lighttree_detail::get_center(lightMem, static_cast<u16>(lightData.type));

				// What happens depends on the light type
				switch(lightData.type) {
					case scene::lights::LightType::POINT_LIGHT: {
						// Always visible, trace for silhouette
						ei::Vec3 fromLight = vertex->get_position() - lightCenter;
						const float lightDist = ei::len(fromLight);
						fromLight /= lightDist;

						if(trace_shadow_silhouette(ei::Ray{ lightCenter, fromLight }, *vertex, lightDist))
							outputBuffer.contribute(coord, RenderTargets::SHADOW_SILHOUETTE, ei::Vec4{ 1.f });
					}	break;
					case scene::lights::LightType::SPOT_LIGHT: {
						// Gotta check first if the vertex is illuminated
						ei::Vec3 fromLight = vertex->get_position() - lightCenter;
						const float lightDist = ei::len(fromLight);
						fromLight /= lightDist;
						const auto& spotLight = as<scene::lights::SpotLight>(lightMem);
						const auto dot = ei::dot(fromLight, spotLight->direction);
						if(ei::abs(ei::dot(fromLight, spotLight->direction)) >= __half2float(spotLight->cosThetaMax)) {
							if(trace_shadow_silhouette(ei::Ray{ lightCenter, fromLight }, *vertex, lightDist))
								outputBuffer.contribute(coord, RenderTargets::SHADOW_SILHOUETTE, ei::Vec4{ 1.f });
						}
					}	break;
				}
			}

			for(u32 lightIdx = 0u; lightIdx < scene.lightTree.dirLights.lightCount; ++lightIdx) {
				auto lightData = get_light_data(lightIdx, scene.lightTree.dirLights);
				const char* lightMem = scene.lightTree.dirLights.memory + lightData.offset;
				const ei::Vec3 lightCenter = scene::lights::lighttree_detail::get_center(lightMem, static_cast<u16>(lightData.type));

				// What happens depends on the light type
				switch(lightData.type) {
					case scene::lights::LightType::DIRECTIONAL_LIGHT: {
						// We have the reverse problem here - no position, but instead a direction is given
						const ei::Vec3 lightPos = vertex->get_position() - scene::MAX_SCENE_SIZE * lightCenter;
						if(trace_shadow_silhouette(ei::Ray{ lightPos, lightCenter }, *vertex, scene::MAX_SCENE_SIZE))
							outputBuffer.contribute(coord, RenderTargets::SHADOW_SILHOUETTE, ei::Vec4{ 1.f });
					}	break;
				}
			}
		}

		++pathLen;
	} while(pathLen < m_params.maxPathLength);
}

bool CpuShadowSilhouettes::trace_shadow_silhouette(const ei::Ray& shadowRay, const PtPathVertex& vertex,
												   const float lightDist) {
	constexpr float DIST_EPSILON = 0.001f;

	const auto firstHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, shadowRay, vertex.get_primitive_id(), lightDist + DIST_EPSILON);
	// TODO: worry about spheres?
	if(firstHit.hitId.instanceId >= 0 && firstHit.hitId != vertex.get_primitive_id()) {
		const ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * firstHit.hitT, shadowRay.direction };

		const auto secondHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, backfaceRay, firstHit.hitId,
																				  lightDist - firstHit.hitT + DIST_EPSILON);
		// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
		if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex.get_primitive_id()
		   && secondHit.hitId.instanceId == firstHit.hitId.instanceId) {
			// Check for silhouette - get the vertex indices of the primitives
			const auto& obj = m_sceneDesc.lods[m_sceneDesc.lodIndices[firstHit.hitId.instanceId]];
			const i32 firstNumVertices = firstHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
			const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
			const i32 firstPrimIndex = firstHit.hitId.primId - (firstHit.hitId.primId < (i32)obj.polygon.numTriangles
																? 0 : (i32)obj.polygon.numTriangles);
			const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
																  ? 0 : (i32)obj.polygon.numTriangles);
			const i32 firstVertOffset = firstHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
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
				const ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (firstHit.hitT + secondHit.hitT), shadowRay.direction };

				const auto thirdHit = scene::accel_struct::first_intersection_scene_lbvh(m_sceneDesc, silhouetteRay, secondHit.hitId,
																						 lightDist - firstHit.hitT - secondHit.hitT + DIST_EPSILON);
				if(thirdHit.hitId == vertex.get_primitive_id()) {
					for(i32 i = 0; i < sharedVertices; ++i) {
						// x86_64 doesn't support atomic_fetch_add for floats FeelsBadMan
						// TODO: proper amount of importance
						atomic_add(m_importanceMap[m_vertexOffsets[firstHit.hitId.instanceId] + edgeIdxFirst[i]], 1.f);
						atomic_add(m_importanceMap[m_vertexOffsets[firstHit.hitId.instanceId] + edgeIdxSecond[i]], 1.f);
					}
					return true;
				} else {
					mAssert(thirdHit.hitId.instanceId >= 0);
					// TODO: store a shadow photon?
				}
			}
		}
	}
	return false;
}

void CpuShadowSilhouettes::reset() {
	this->m_reset = true;
}

void CpuShadowSilhouettes::pt_sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
						   const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
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
			auto nee = connect(scene.lightTree, 0, 1, neeSeed,
							   vertex->get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex->evaluate(nee.direction, scene.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
					scene, { vertex->get_position() , nee.direction },
					vertex->get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
											value.cosOut, radiance * mis);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		math::DirectionSample lastDir;
		if(!walk(scene, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(scene.lightTree.background, lastDir.direction);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
					background.value *= mis;
					outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}

		if(pathLen == 0 && m_params.importanceIterations > 0 && outputBuffer.is_target_enabled(RenderTargets::IMPORTANCE)) {
			outputBuffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ compute_importance(vertex->get_primitive_id()) });
		}

		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= m_params.maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(scene.lightTree, vertex->get_primitive_id(),
												  vertex->get_surface_params(),
												  lastPosition, scene::lights::guide_flux);
				float mis = pathLen == 1 ? 1.0f
					: 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				emission *= mis;
			}
			outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
									vertex->get_normal(), vertex->get_albedo());
		}
	} while(pathLen < m_params.maxPathLength);
}

void CpuShadowSilhouettes::compute_max_importance() {
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

void CpuShadowSilhouettes::display_importance(RenderBuffer<Device::CPU>& buffer) {
	const u32 NUM_PIXELS = buffer.get_height() * buffer.get_width();
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < (int)NUM_PIXELS; ++pixel) {
		const ei::IVec2 coord{ pixel % buffer.get_width(), pixel / buffer.get_width() };
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
			buffer.contribute(coord, RenderTargets::IMPORTANCE, ei::Vec4{ compute_importance(vertex->get_primitive_id()) });
	}

}

float CpuShadowSilhouettes::compute_importance(const scene::PrimitiveHandle& hitId) {
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

void CpuShadowSilhouettes::initialize_importance_map() {
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
		m_importanceMap[i].store(1.f);
}

void CpuShadowSilhouettes::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

void CpuShadowSilhouettes::load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) {
	if(scene != m_currentScene) {
		m_currentScene = scene;
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, resolution);
		m_reset = true;
	}
}

} // namespace mufflon::renderer