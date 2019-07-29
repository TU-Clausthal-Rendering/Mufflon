#pragma once

#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <utility>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

using namespace scene::lights;

namespace {

CUDA_FUNCTION std::array<ei::Vec3, 3u> get_world_triangle(const scene::SceneDescriptor<CURRENT_DEV>& scene,
														  const scene::PrimitiveHandle& hitId) {
	const auto& polygon = scene.lods[scene.lodIndices[hitId.instanceId]].polygon;
	return std::array<ei::Vec3, 3u>{{
		ei::transform(polygon.vertices[3u * hitId.primId + 0], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[3u * hitId.primId + 1], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[3u * hitId.primId + 2], scene.worldToInstance[hitId.instanceId])
	}};
}

CUDA_FUNCTION std::array<ei::Vec3, 4u> get_world_quad(const scene::SceneDescriptor<CURRENT_DEV>& scene,
													  const scene::PrimitiveHandle& hitId) {
	const auto& polygon = scene.lods[scene.lodIndices[hitId.instanceId]].polygon;
	const auto offset = 3u * polygon.numTriangles;
	const auto quadId = hitId.primId - polygon.numTriangles;
	return std::array<ei::Vec3, 4u>{{
		ei::transform(polygon.vertices[offset + 4u * quadId + 0], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 1], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 2], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 3], scene.worldToInstance[hitId.instanceId]),
	}};
}

CUDA_FUNCTION std::array<ei::Vec3, 3u> get_world_light_triangle_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																	  const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(lightMemory);
	// Returns v1 - v0, v2 - v0, v1 - v2
	return std::array<ei::Vec3, 3u>{{
		light.posV[1], light.posV[2], light.posV[1] - light.posV[2]
	}};
}

CUDA_FUNCTION std::array<ei::Vec3, 4u> get_world_light_quad_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																  const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(lightMemory);
	// Return v3-v0, v1-v0, v2-v1, v2-v3
	return std::array<ei::Vec3, 4u>{{
		light.posV[1], light.posV[2], light.posV[3] + light.posV[1], light.posV[3] + light.posV[2]
	}};
}

template < class T, std::size_t N, class Functor >
CUDA_FUNCTION std::array<T, N> apply(const std::array<T, N>& values, Functor&& functor) {
	std::array<T, N> res;
	for(std::size_t i = 0u; i < N; ++i)
		res[i] = functor(values[i]);
	return res;
}

template < class T, std::size_t N >
CUDA_FUNCTION decltype(ei::len(std::declval<T>())) get_shortest_length(const std::array<T, N>& values) {
	auto shortestLength = ei::len(values[0u]);
	for(std::size_t i = 1u; i < N; ++i) {
		const auto length = ei::len(values[i]);
		if(length < shortestLength)
			shortestLength = length;
	}
	return shortestLength;
}

CUDA_FUNCTION constexpr float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

CUDA_FUNCTION void record_silhouette_vertex_contribution(Importances<CURRENT_DEV>* importances,
														 DeviceImportanceSums<CURRENT_DEV>& sums,
														 const u32 vertexIndex, const float importance) {
	// Reminder: local index will refer to the decimated mesh
	cuda::atomic_add<CURRENT_DEV>(importances[vertexIndex].viewImportance, importance);
	cuda::atomic_add<CURRENT_DEV>(sums.shadowSilhouetteImportance, importance);
}

CUDA_FUNCTION void record_shadow(DeviceImportanceSums<CURRENT_DEV>& sums, const float irradiance) {
	cuda::atomic_add<CURRENT_DEV>(sums.shadowImportance, irradiance);
}

CUDA_FUNCTION void record_direct_hit(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
									 const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint,
									 const float cosAngle, const float sharpness) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(cosAngle))
		cuda::atomic_add<CURRENT_DEV>(importances[min].viewImportance, sharpness * (1.f - ei::abs(cosAngle)));
}

CUDA_FUNCTION void record_direct_irradiance(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
											const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint, const float irradiance) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(irradiance)) {
		cuda::atomic_add<CURRENT_DEV>(importances[min].irradiance, irradiance);
		cuda::atomic_add<CURRENT_DEV>(importances[min].hitCounter, 1u);
	}
}

CUDA_FUNCTION void record_indirect_irradiance(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
											  const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint, const float irradiance) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(irradiance))
		cuda::atomic_add<CURRENT_DEV>(importances[min].irradiance, irradiance);
}

CUDA_FUNCTION bool trace_shadow(const scene::SceneDescriptor<CURRENT_DEV>& scene, DeviceImportanceSums<CURRENT_DEV>* sums,
								const ei::Ray& shadowRay, SilPathVertex& vertex, const float importance,
								const scene::PrimitiveHandle& shadowHitId, const float lightDistance,
								const float firstShadowDistance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(scene, backfaceRay, vertex.get_geometric_normal(),
																   lightDistance - firstShadowDistance + DIST_EPSILON);
	if(secondHit.hitId.instanceId < 0 || secondHit.hitId == vertex.get_primitive_id())
		return false;

	ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (firstShadowDistance + secondHit.distance), shadowRay.direction };
	const auto thirdHit = scene::accel_struct::first_intersection(scene, silhouetteRay, secondHit.normal,
																  lightDistance - firstShadowDistance - secondHit.distance + DIST_EPSILON);
	if(thirdHit.hitId == vertex.get_primitive_id()) {
		const auto lodIdx = scene.lodIndices[shadowHitId.instanceId];
		record_shadow(sums[lodIdx], importance);

		// Also check for silhouette interaction here
		if(secondHit.hitId.instanceId == vertex.get_primitive_id().instanceId) {
			const auto& obj = scene.lods[scene.lodIndices[secondHit.hitId.instanceId]];
			const i32 firstNumVertices = shadowHitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
			const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
			const i32 firstPrimIndex = shadowHitId.primId - (shadowHitId.primId < (i32)obj.polygon.numTriangles
																		? 0 : (i32)obj.polygon.numTriangles);
			const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
																  ? 0 : (i32)obj.polygon.numTriangles);
			const i32 firstVertOffset = shadowHitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
			const i32 secondVertOffset = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;

			// Check if we have "shared" vertices: cannot do it by index, since they might be
			// split vertices, but instead need to go by proximity
			i32 sharedVertices = 0;
			for(i32 i0 = 0; i0 < firstNumVertices; ++i0) {
				for(i32 i1 = 0; i1 < secondNumVertices; ++i1) {
					const i32 idx0 = obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + i0];
					const i32 idx1 = obj.polygon.vertexIndices[secondVertOffset + secondNumVertices * secondPrimIndex + i1];
					const ei::Vec3& p0 = obj.polygon.vertices[idx0];
					const ei::Vec3& p1 = obj.polygon.vertices[idx1];
					if(idx0 == idx1 || p0 == p1) {
						vertex.ext().silhouetteVerticesFirst[sharedVertices] = idx0;
						vertex.ext().silhouetteVerticesFirst[sharedVertices] = idx1;
						// TODO: deal with boundaries/split vertices
						++sharedVertices;
					}
					if(sharedVertices >= 2)
						break;
				}
			}
			if(sharedVertices >= 1)
				vertex.ext().shadowInstanceId = secondHit.hitId.instanceId;
		}

		return true;
	}
	return false;
}

} // namespace

CUDA_FUNCTION void sample_importance(renderer::RenderBuffer<CURRENT_DEV>& outputBuffer,
									 const scene::SceneDescriptor<CURRENT_DEV>& scene,
									 const SilhouetteParameters& params,
									 const Pixel& coord, math::Rng& rng,
									 Importances<CURRENT_DEV>** importances,
									 DeviceImportanceSums<CURRENT_DEV>* sums) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	// We gotta keep track of our vertices
	// TODO: flexible length!
#ifdef __CUDA_ARCH__
	SilPathVertex vertices[16];
#else // __CUDA_ARCH__
	thread_local SilPathVertex vertices[16];
#endif // __CUDA_ARCH__
	//thread_local std::vector<SilPathVertex> vertices(std::max(2, params.maxPathLength + 1));
	//vertices.clear();
	// Create a start for the path
	(void)SilPathVertex::create_camera(&vertices[0], &vertices[0], scene.camera.get(), coord, rng.next());

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
		if(pathLen > 0 && pathLen + 1 <= params.maxPathLength) {
			u64 neeSeed = rng.next();
			math::RndSet2 neeRnd = rng.next();
			u32 lightIndex;
			u32 lightOffset;
			scene::lights::LightType lightType;
			auto nee = scene::lights::connect(scene, 0, 1, neeSeed, vertices[pathLen].get_position(), neeRnd,
											  &lightIndex, &lightType, &lightOffset);
			Pixel projCoord;
			auto value = vertices[pathLen].evaluate(nee.dir.direction, scene.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			// TODO: use multiple NEEs
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				ei::Ray shadowRay{ nee.position, -nee.dir.direction };

				auto shadowHit = scene::accel_struct::first_intersection(scene, shadowRay,
																		 vertices[pathLen].get_geometric_normal(), nee.dist);
				const float firstShadowDistance = shadowHit.distance;
				AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

				const float weightedIrradianceLuminance = get_luminance(throughput.weight * irradiance) *(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				if(shadowHit.hitId.instanceId < 0) {
					mAssert(!isnan(mis));
					// Save the radiance for the later indirect lighting computation
					// Compute how much radiance arrives at the previous vertex from the direct illumination
					// Add the importance

					const auto& hitId = vertices[pathLen].get_primitive_id();
					const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
					const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
					record_direct_irradiance(lod.polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId,
											 numVertices, vertices[pathLen].get_position(), params.lightWeight * weightedIrradianceLuminance);
				} else {
					// The "importance" we add for general shadow casting (not part of a silhouette)
					// scales inversely with the smallest light source size in the plane defined by
					// the shadow ray. As an approximation of this we use the shortest projected
					// border for triangles and quads.

					float lengthEstimate = 0.f;	// Valid for point and directional lights
					switch(lightType) {
						case scene::lights::LightType::AREA_LIGHT_TRIANGLE: {
							// Compute the projected sides and choose the shortest
							const auto planeNormal = nee.dir.direction;
							const auto sides = get_world_light_triangle_sides(scene, lightOffset);
							const auto projected = apply(sides, [planeNormal] __host__ __device__ (const ei::Vec3& side) {
								return side - ei::dot(side, planeNormal) * planeNormal;
							});
							lengthEstimate = get_shortest_length(projected);
						}	break;
						case scene::lights::LightType::AREA_LIGHT_QUAD: {
							// Compute the projected sides and choose the shortest
							const auto planeNormal = nee.dir.direction;
							const auto sides = get_world_light_quad_sides(scene, lightOffset);
							const auto projected = apply(sides, [planeNormal](const ei::Vec3& side) {
								return side - ei::dot(side, planeNormal) * planeNormal;
							});
							lengthEstimate = get_shortest_length(projected);
						}	break;
						case scene::lights::LightType::AREA_LIGHT_SPHERE: {
							const auto& light = *reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(scene.lightTree.posLights.memory + lightOffset);
							lengthEstimate = light.radius;
						}	break;
						case scene::lights::LightType::ENVMAP_LIGHT:
							// TODO: pre-processed list of light areas!
							break;
						default:
							break;
					}

					// Compute the shadow size from intercept theorem
					const float distShadowToOccluder = nee.dist - shadowHit.distance;
					const float projectedLightSize = lengthEstimate * distShadowToOccluder / shadowHit.distance;
					// Scale the irradiance with the predicted shadow size
					const float shadowWeightedIrradiance = 1.f / (1.f + projectedLightSize) * weightedIrradianceLuminance;

					//m_decimaters[scene.lodIndices[shadowHit.hitId.instanceId]]->record_shadow(get_luminance(throughput.weight * irradiance));
					trace_shadow(scene, sums, shadowRay, vertices[pathLen], shadowWeightedIrradiance,
								 shadowHit.hitId, nee.dist, firstShadowDistance);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		VertexSample sample;
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertices[pathLen], rnd, rndRoulette, false, throughput, vertices[pathLen + 1], sample) == WalkResult::CANCEL)
			break;

		// Terminate on background
		if(vertices[pathLen + 1].is_end_point()) break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().updateBxdf(sample, throughput);

		// Don't update sharpness for camera vertex
		if(pathLen > 0) {
			const ei::Vec3 bxdf = sample.throughput * (float)sample.pdf.forw;
			const float bxdfLum = get_luminance(bxdf);
			if(isnan(bxdfLum))
				return;
			sharpness *= 2.f / (1.f + ei::exp(-bxdfLum / params.sharpnessFactor)) - 1.f;
		}

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;

		record_direct_hit(lod.polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId, numVertices, vertices[pathLen].get_position(),
						  -ei::dot(vertices[pathLen + 1].get_incident_direction(), vertices[pathLen + 1].get_normal()),
						  params.viewWeight * sharpness);
		++pathLen;
	} while(pathLen < params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{ 0.f };
	for(int p = pathLen - 2; p >= 1; --p) {
		accumRadiance = vertices[p].ext().throughput * accumRadiance + (vertices[p + 1].ext().shadowInstanceId == 0 ?
																		vertices[p + 1].ext().pathRadiance : ei::Vec3{ 0.f });
		const ei::Vec3 irradiance = vertices[p].ext().outCos * accumRadiance;

		const auto& hitId = vertices[p].get_primitive_id();
		const auto* lod = &scene.lods[scene.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;

		const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
		record_indirect_irradiance(lod->polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId,
								   numVertices, vertices[p].get_position(), params.lightWeight * importance);
		// TODO: store accumulated sharpness
		// Check if it is sensible to keep shadow silhouettes intact
		// TODO: replace threshold with something sensible
		if(p == 1 && vertices[p].ext().shadowInstanceId >= 0) {
			const float indirectLuminance = get_luminance(accumRadiance);
			const float totalLuminance = get_luminance(vertices[p].ext().pathRadiance) + indirectLuminance;
			const float ratio = totalLuminance / indirectLuminance - 1.f;
			if(ratio > 0.02f) {
				constexpr float FACTOR = 2'000.f;

				// TODO: proper factor!
				const auto lodIdx = scene.lodIndices[vertices[p].ext().shadowInstanceId];
				for(i32 i = 0; i < 2; ++i) {
					record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx],
															vertices[p].ext().silhouetteVerticesFirst[i],
															params.shadowSilhouetteWeight * FACTOR * (totalLuminance - indirectLuminance));
					if(vertices[p].ext().silhouetteVerticesFirst[i] != vertices[p].ext().silhouetteVerticesSecond[i])
						record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx],
															  vertices[p].ext().silhouetteVerticesSecond[i],
															  params.shadowSilhouetteWeight * FACTOR * (totalLuminance - indirectLuminance));
				}
			}
		}
	}
}

CUDA_FUNCTION void sample_vis_importance(renderer::RenderBuffer<CURRENT_DEV>& outputBuffer,
										 const scene::SceneDescriptor<CURRENT_DEV>& scene,
										 const Pixel& coord, math::Rng& rng,
										 Importances<CURRENT_DEV>** importances,
										 const float maxImportance) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample) == WalkResult::HIT) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		const auto& polygon = scene.lods[lodIdx].polygon;
		const u32 vertexCount = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
		const u32 primIdx = vertexCount == 3u ? hitId.primId : (hitId.primId - polygon.numTriangles);

		float importance = 0.f;
		float distSqrSum = 0.f;
		for(u32 i = 0u; i < vertexCount; ++i)
			distSqrSum += ei::lensq(hitpoint - polygon.vertices[polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i]]);
		for(u32 i = 0u; i < vertexCount; ++i) {
			const auto vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i];
			const float distSqr = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
			importance += importances[lodIdx][vertexIndex].viewImportance;
		}

		outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec3{ importance / maxImportance });
	}
}

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt