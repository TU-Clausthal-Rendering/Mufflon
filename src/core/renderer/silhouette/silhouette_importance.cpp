#include "cpu_silhouette.hpp"
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

struct ShadowCandidate {
	ei::Ray shadowRay;
	scene::PrimitiveHandle hitId;
	float hitT;
};

/**
 * Gathering the importance is a multi-step process:
 * First, we perform a regular path tracing sample, ie. we trace a path, evaluate BRDFs, perform next-event estimation...
 * We then trace the path backwards to evaluate how "sharp" the feature of the path is via the BRDF.
 * When the BRDF is suitably sharp, the possibility of shadow silhouettes 
 */
void CpuShadowSilhouettes::importance_sample_weighted(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
													  const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	// We gotta remember all vertices unfortunately...
	u8 allVertexBuffer[256u*16u];// TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex* vertex = as<PtPathVertex>(allVertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");

	// Remember the direct illumination of each path segment
	if(m_params.maxPathLength > 16u)
		throw std::runtime_error("PT path length too long (> 16)");
	float viewPathDirectLuminance[16u];
	float viewPathIndirectLuminance[16u];
	float viewPathBxdf[16u];
	u32 connectedLightIndices[16u];
	ShadowCandidate shadowCandidates[16u];

	// Start with a BxDF of 1
	viewPathBxdf[0] = 1.f;

	int pathLen = 0;
	do {
		// Connection to light source
		if(pathLen > 0 && pathLen + 1 <= m_params.maxPathLength) {
			u64 neeSeed = m_rngs[pixel].next();
			math::RndSet2 neeRnd = m_rngs[pixel].next();
			auto nee = connect(scene.lightTree, 0, 1, neeSeed,
							   vertex->get_position(), m_currentScene->get_bounding_box(),
							   neeRnd, scene::lights::guide_flux,
							   &connectedLightIndices[pathLen]);
			auto value = vertex->evaluate(nee.direction, scene.media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				auto shadowHit = scene::accel_struct::first_intersection_scene_lbvh<Device::CPU>(
					scene, ei::Ray{ nee.lightPoint, -nee.direction }, vertex->get_primitive_id(), nee.dist);
				shadowCandidates[pathLen] = { ei::Ray{ nee.lightPoint, -nee.direction }, shadowHit.hitId, shadowHit.hitT };

				AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				mAssert(!isnan(mis));
				viewPathDirectLuminance[pathLen] = get_luminance(radiance * throughput.weight) * mis * value.cosOut;
			} else {
				viewPathDirectLuminance[pathLen] = 0;
			}
		}
		
		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
		PtPathVertex* outVertex = as<PtPathVertex>(&allVertexBuffer[256*(pathLen+1)]);
		math::DirectionSample lastDir;
		if(!walk(scene, *vertex, rnd, -1.0f, false, throughput, outVertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(scene.lightTree.background, lastDir.direction);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
					background.value *= mis;
					viewPathIndirectLuminance[pathLen] = get_luminance(background.value * throughput.weight);
				}
			}
			break;
		}
		vertex = outVertex;

		// Reconstruct the BxDF for later use
		viewPathBxdf[pathLen + 1] = viewPathBxdf[pathLen] * get_luminance(throughput.weight) * (float)lastDir.pdf / ei::dot(vertex->get_incident_direction(), lastDir.direction);

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
			viewPathIndirectLuminance[pathLen] = get_luminance(emission * throughput.weight);
		}

		++pathLen;
	} while(pathLen < m_params.maxPathLength);

	float cumulativeRadiance = 0.f;
	// Now we can go backwards over the path and decide if it is sensible to shoot silhouette rays
	for(int p = (m_params.enableIndirectImportance ? 0 : std::max(0, pathLen - 2)); p < pathLen - 1; ++p) {
		const int pathIndex = pathLen - p - 1;

		PtPathVertex* currentVertex = as<PtPathVertex>(&allVertexBuffer[pathIndex * 256u]);
		// TODO: Use the last BRDF as an indicator for termination
		//if(viewPathBxdf[pathIndex] < 0.875f) break;

		cumulativeRadiance += viewPathIndirectLuminance[pathIndex];
		// Check if we're shadowed by something
		if(shadowCandidates[pathIndex].hitId.instanceId >= 0) {
			// Check how big the difference between direct light and no direct light would be
			float luminanceRatio;
			if(cumulativeRadiance == 0.f)
				luminanceRatio = std::numeric_limits<float>::infinity();
			else
				luminanceRatio = viewPathDirectLuminance[pathIndex] / cumulativeRadiance;

			// We only do shadow silhouettes if the ratio is sufficiently big (Weber-Fechner-Law)
			// TODO: how to prperly bring in the BxDF?
			if(viewPathBxdf[pathIndex] * luminanceRatio > m_params.directIndirectRatio && m_params.enableSilhouetteImportance) {
				// TODO: store the hit?
				(void)trace_shadow_silhouette_shadow(shadowCandidates[pathIndex].shadowRay, *currentVertex,
													 shadowCandidates[pathIndex].hitId, shadowCandidates[pathIndex].hitT,
													 ei::len(shadowCandidates[pathIndex].shadowRay.origin - currentVertex->get_position()),
													 viewPathDirectLuminance[pathIndex]);
			}
		} else {
			cumulativeRadiance += viewPathDirectLuminance[pathIndex];
		}

		if(m_params.enableViewImportance) {
			// Add importance: what is this vertex' contribution to the sensor's perception
			const auto& hitId = currentVertex->get_primitive_id();
			const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
			const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
			const u32 vertexOffset = hitId.primId < (i32)lod.polygon.numTriangles ? 0u : 3u * lod.polygon.numTriangles;
			for(u32 i = 0u; i < numVertices; ++i) {
				const u32 vertexIndex = lod.polygon.vertexIndices[vertexOffset + numVertices * hitId.primId + i];
				const u32 index = m_vertexOffsets[hitId.instanceId] + vertexIndex;

				atomic_add(m_importanceMap[index], cumulativeRadiance);
			}
		}
	}
}


bool CpuShadowSilhouettes::trace_shadow_silhouette_shadow(const ei::Ray& shadowRay, const PtPathVertex& vertex,
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
					// TODO: proper amount of importance
					logWarning(importance);
					atomic_add(m_importanceMap[m_vertexOffsets[firstHit.instanceId] + edgeIdxFirst[i]], importance);
					atomic_add(m_importanceMap[m_vertexOffsets[firstHit.instanceId] + edgeIdxSecond[i]], importance);
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

} // namespace mufflon::renderer