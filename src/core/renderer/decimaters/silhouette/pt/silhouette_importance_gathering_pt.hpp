#pragma once

#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include "core/export/core_api.h"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <ei/3dintersection.hpp>
#include <utility>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

using namespace scene::lights;

namespace {

// Checks whether a ray intersects a coplanar line segment and returns the intersection point
inline CUDA_FUNCTION auto intersects_coplanar(const ei::Segment& ray, const ei::Segment& segment) {
	constexpr float ERROR_DELTA = 0.00125f;
	struct Result {
		bool intersects;
		ei::Vec3 point;
	};
	const ei::Vec3 originDiff = segment.a - ray.a;
	const ei::Vec3 segmentDir = segment.b - segment.a;
	const ei::Vec3 rayDir = ray.b - ray.a;
	// No need to check for co-planarity
	const ei::Vec3 normal = ei::cross(rayDir, segmentDir);
	const float s = ei::dot(ei::cross(originDiff, segmentDir), normal) / ei::lensq(normal);
	const ei::Vec3 intersection = ray.a + s * rayDir;
	// Check if we're within the segment
	const float segmentLenSqr = ei::lensq(segment.a - segment.b);
	const float intersectionDistSq = ei::lensq(intersection - segment.a) + ei::lensq(intersection - segment.b);
	if(intersectionDistSq <= segmentLenSqr + ERROR_DELTA)
		return Result{ true , intersection };
	return Result{ false, ei::Vec3{} };
}

inline CUDA_FUNCTION ei::Triangle get_world_triangle(const scene::SceneDescriptor<CURRENT_DEV>& scene,
											  const scene::PrimitiveHandle& hitId) {
	const auto& polygon = scene.lods[scene.lodIndices[hitId.instanceId]].polygon;
	return ei::Triangle{
		ei::transform(polygon.vertices[3u * hitId.primId + 0], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[3u * hitId.primId + 1], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[3u * hitId.primId + 2], scene.worldToInstance[hitId.instanceId])
	};
}

inline CUDA_FUNCTION ei::Tetrahedron get_world_quad(const scene::SceneDescriptor<CURRENT_DEV>& scene,
											 const scene::PrimitiveHandle& hitId) {
	const auto& polygon = scene.lods[scene.lodIndices[hitId.instanceId]].polygon;
	const auto offset = 3u * polygon.numTriangles;
	const auto quadId = hitId.primId - polygon.numTriangles;
	return ei::Tetrahedron{
		ei::transform(polygon.vertices[offset + 4u * quadId + 0], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 1], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 2], scene.worldToInstance[hitId.instanceId]),
		ei::transform(polygon.vertices[offset + 4u * quadId + 3], scene.worldToInstance[hitId.instanceId]),
	};
}

inline CUDA_FUNCTION ei::Triangle get_world_light_triangle(const scene::SceneDescriptor<CURRENT_DEV>& scene,
													const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(lightMemory);
	// Returns v0, v1, v2
	return ei::Triangle{
		light.posV[0], light.posV[1] + light.posV[0], light.posV[2] + light.posV[0]
	};
}

inline CUDA_FUNCTION ei::Tetrahedron get_world_light_quad(const scene::SceneDescriptor<CURRENT_DEV>& scene,
												   const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(lightMemory);
	// Return v0, v1, v2, v3
	const ei::Vec3 v3 = light.posV[1] + light.posV[0];
	return ei::Tetrahedron{
		light.posV[0], light.posV[2] + light.posV[0], light.posV[3] + light.posV[2] + v3, v3
	};
}

inline CUDA_FUNCTION ei::Triangle get_world_light_triangle_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																	  const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightTriangle<CURRENT_DEV>*>(lightMemory);
	// Returns v1 - v0, v2 - v0, v1 - v2
	return ei::Triangle{ light.posV[1], light.posV[2],
						 light.posV[1] - light.posV[2] };
}

inline CUDA_FUNCTION ei::Tetrahedron get_world_light_quad_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																  const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const AreaLightQuad<CURRENT_DEV>*>(lightMemory);
	// Return v3-v0, v1-v0, v2-v1, v2-v3
	return ei::Tetrahedron{ light.posV[1], light.posV[2], light.posV[3] + light.posV[1],
							light.posV[3] + light.posV[2] };
}

inline CUDA_FUNCTION ei::Vec3 project_onto_plane(const ei::Vec3& point, const ei::Plane& plane) {
	const float distToPlane = ei::dot(plane.n, point) + plane.d;
	return point - plane.n * distToPlane;
}
inline CUDA_FUNCTION ei::Triangle project_onto_plane(const ei::Triangle& tri, const ei::Plane& plane) {
	return ei::Triangle{
		project_onto_plane(tri.v0, plane),
		project_onto_plane(tri.v1, plane),
		project_onto_plane(tri.v2, plane)
	};
}
inline CUDA_FUNCTION ei::Tetrahedron project_onto_plane(const ei::Tetrahedron& quad, const ei::Plane& plane) {
	return ei::Tetrahedron{
		project_onto_plane(quad.v0, plane),
		project_onto_plane(quad.v1, plane),
		project_onto_plane(quad.v2, plane),
		project_onto_plane(quad.v3, plane)
	};
}
// Takes a triangle, a plane with a point inside that triangle, and an origin point.
// It then projects the triangle points onto the plane, computes the intersection points
// of the ray defined by the intersection point and plane normal, and returns the distance
// between the intersection points.
inline CUDA_FUNCTION float intersect_triangle_borders(const ei::Triangle& triangle, const ei::Plane& plane,
											   const ei::Segment& planarSegment) {
	// First project the triangle points onto the plane
	// TODO: "proper" would be a perspective projection, which is more important the larger
	// the triangle is
	const ei::Triangle projectedTriangle = project_onto_plane(triangle, plane);

	// Determine the two intersections with triangle sides
	const auto v0v1 = intersects_coplanar(planarSegment, ei::Segment{ projectedTriangle.v0, projectedTriangle.v1 });
	const auto v0v2 = intersects_coplanar(planarSegment, ei::Segment{ projectedTriangle.v0, projectedTriangle.v2 });
	const auto v1v2 = intersects_coplanar(planarSegment, ei::Segment{ projectedTriangle.v1, projectedTriangle.v2 });
	ei::distance(ei::Segment{}, ei::Segment{});

	if(v0v1.intersects) {
		if(v0v2.intersects)
			return ei::len(v0v1.point - v0v2.point);
		else if(v1v2.intersects)
			return ei::len(v0v1.point - v1v2.point);
		else
			mAssertMsg(false, "This shouldn't happen!");
	} else if(v0v2.intersects) {
		if(v1v2.intersects)
			return ei::len(v0v2.point - v1v2.point);
		else
			mAssertMsg(false, "This shouldn't happen!");
	} else {
		mAssertMsg(false, "This shouldn't happen!");
	}
	return 0.f;
}

// Takes a quad, a plane with a point inside that triangle, and an origin point.
// It then projects the quad points onto the plane, computes the intersection points
// of the ray defined by the intersection point and plane normal, and returns the distance
// between the intersection points.
inline CUDA_FUNCTION float intersect_quad_borders(const ei::Tetrahedron& quad, const ei::Plane& plane,
										   const ei::Segment& planarSegment) {
	// First project the quad points onto the plane
	// TODO: "proper" would be a perspective projection, which is more important the larger
	// the quad is
	const ei::Tetrahedron projectedQuad = project_onto_plane(quad, plane);

	// Determine the two intersections with triangle sides
	const auto v0v1 = intersects_coplanar(planarSegment, ei::Segment{ projectedQuad.v0, projectedQuad.v1 });
	const auto v0v3 = intersects_coplanar(planarSegment, ei::Segment{ projectedQuad.v0, projectedQuad.v3 });
	const auto v2v1 = intersects_coplanar(planarSegment, ei::Segment{ projectedQuad.v2, projectedQuad.v1 });
	const auto v2v3 = intersects_coplanar(planarSegment, ei::Segment{ projectedQuad.v2, projectedQuad.v3 });

	if(v0v1.intersects) {
		if(v0v3.intersects)
			return ei::len(v0v1.point - v0v3.point);
		else if(v2v1.intersects)
			return ei::len(v0v1.point - v2v1.point);
		else if(v2v3.intersects)
			return ei::len(v0v1.point - v2v3.point);
		else
			mAssertMsg(false, "This shouldn't happen!");
	} else if(v0v3.intersects) {
		if(v2v1.intersects)
			return ei::len(v0v3.point - v2v1.point);
		else if(v2v3.intersects)
			return ei::len(v0v3.point - v2v3.point);
		else
			mAssertMsg(false, "This shouldn't happen!");
	} else if(v2v1.intersects) {
		if(v2v3.intersects)
			return ei::len(v2v1.point - v2v3.point);
		else
			mAssertMsg(false, "This shouldn't happen!");
	} else {
		mAssertMsg(false, "This shouldn't happen!");
	}
	return 0.f;
}

// Takes a triangle type, but the individual vertices contain lengths
inline CUDA_FUNCTION constexpr float get_shortest_length(const ei::Triangle& tri) {
	return ei::min(ei::min(ei::len(tri.v0), ei::len(tri.v1)), ei::len(tri.v2));
}
// Takes a quad type, but the individual vertices contain lengths
inline CUDA_FUNCTION constexpr float get_shortest_length(const ei::Tetrahedron& quad) {
	return ei::min(ei::min(ei::min(ei::len(quad.v0), ei::len(quad.v1)), ei::len(quad.v2)), ei::len(quad.v3));
}

inline CUDA_FUNCTION float estimate_silhouette_light_size(const scene::SceneDescriptor<CURRENT_DEV>& scene,
												   const LightType lightType, const u32 lightOffset,
												   const ei::Ray& shadowRay, const scene::PrimitiveHandle shadowHitId,
												   const SilPathVertex& vertex, const ei::Plane& neePlane,
												   const float occluderLightDistance, const float occluderShadowDistance) {
	float size = 0.f;
	switch(lightType) {
		case scene::lights::LightType::AREA_LIGHT_TRIANGLE: {
			const ei::Triangle tri = get_world_light_triangle(scene, lightOffset);
			// Project the silhouette edge onto the light plane
			const auto& polygon = scene.lods[scene.lodIndices[shadowHitId.instanceId]].polygon;
			mAssert(vertex.ext().silhouetteVerticesFirst[0] >= 0 && static_cast<u32>(vertex.ext().silhouetteVerticesFirst[0]) < polygon.numVertices);
			mAssert(vertex.ext().silhouetteVerticesFirst[1] >= 0 && static_cast<u32>(vertex.ext().silhouetteVerticesFirst[1]) < polygon.numVertices);
			const auto instToWorld = scene.compute_instance_to_world_transformation(shadowHitId.instanceId);
			const ei::Vec3 a = project_onto_plane(ei::transform(polygon.vertices[vertex.ext().silhouetteVerticesFirst[0]],
																instToWorld), neePlane);
			const ei::Vec3 b = project_onto_plane(ei::transform(polygon.vertices[vertex.ext().silhouetteVerticesFirst[1]],
																instToWorld), neePlane);
			// Compute the 90° rotated vector
			const ei::Vec3 rotatedB = a + ei::cross(neePlane.n, b - a);

			size = intersect_triangle_borders(tri, neePlane, ei::Segment{ a, rotatedB });
		}	break;
		case scene::lights::LightType::AREA_LIGHT_QUAD: {
			const ei::Tetrahedron quad = get_world_light_quad(scene, lightOffset);
			// Project the silhouette edge onto the light plane
			const auto& polygon = scene.lods[scene.lodIndices[shadowHitId.instanceId]].polygon;
			mAssert(vertex.ext().silhouetteVerticesFirst[0] >= 0 && static_cast<u32>(vertex.ext().silhouetteVerticesFirst[0]) < polygon.numVertices);
			mAssert(vertex.ext().silhouetteVerticesFirst[1] >= 0 && static_cast<u32>(vertex.ext().silhouetteVerticesFirst[1]) < polygon.numVertices);
			const auto instToWorld = scene.compute_instance_to_world_transformation(shadowHitId.instanceId);
			const ei::Vec3 a = project_onto_plane(ei::transform(polygon.vertices[vertex.ext().silhouetteVerticesFirst[0]],
																instToWorld),
												  neePlane);
			const ei::Vec3 b = project_onto_plane(ei::transform(polygon.vertices[vertex.ext().silhouetteVerticesFirst[1]],
																instToWorld),
												  neePlane);
			// Compute the 90° rotated vector
			// TODO: end-points of silhouette edge may lie outside of light primitive!
			// TODO: they can both be outside!
			const ei::Vec3 rotatedB = a + ei::cross(neePlane.n, b - a);

			size = intersect_quad_borders(quad, neePlane, ei::Segment{ a, rotatedB });
		}	break;
		case scene::lights::LightType::AREA_LIGHT_SPHERE: {
			const auto& light = *reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(scene.lightTree.posLights.memory + lightOffset);
			size = light.radius;
		}	break;
		case scene::lights::LightType::ENVMAP_LIGHT:
			// TODO: pre-processed list of light areas!
			break;
		default:
			break;
	}

	// Compute the shadow size from intercept theorem
	return size * occluderShadowDistance / occluderLightDistance;
}

inline CUDA_FUNCTION float estimate_shadow_light_size(const scene::SceneDescriptor<CURRENT_DEV>& scene,
											   const LightType lightType, const u32 lightOffset,
											   const ei::Ray& shadowRay, const SilPathVertex& vertex,
											   const ei::Plane& neePlane, const float occluderLightDistance,
											   const float occluderShadowDistance) {
	float size = 0.f;
	switch(lightType) {
		case scene::lights::LightType::AREA_LIGHT_TRIANGLE: {
			// Compute the projected sides and choose the shortest
			const auto sides = get_world_light_triangle_sides(scene, lightOffset);
			const auto projected = project_onto_plane(sides, neePlane);
			size = get_shortest_length(projected);
		}	break;
		case scene::lights::LightType::AREA_LIGHT_QUAD: {
			// Compute the projected sides and choose the shortest
			const auto sides = get_world_light_quad_sides(scene, lightOffset);
			const auto projected = project_onto_plane(sides, neePlane);
			size = get_shortest_length(projected);
		}	break;
		case scene::lights::LightType::AREA_LIGHT_SPHERE: {
			const auto& light = *reinterpret_cast<const AreaLightSphere<CURRENT_DEV>*>(scene.lightTree.posLights.memory + lightOffset);
			size = light.radius;
		}	break;
		case scene::lights::LightType::ENVMAP_LIGHT:
			// TODO: pre-processed list of light areas!
			break;
		default:
			break;
	}

	// Compute the shadow size from intercept theorem
	return size * occluderShadowDistance / occluderLightDistance;
}

inline CUDA_FUNCTION constexpr float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

inline CUDA_FUNCTION void record_silhouette_vertex_contribution(Importances<CURRENT_DEV>* importances,
														 DeviceImportanceSums<CURRENT_DEV>& sums,
														 const u32 vertexIndex, const float importance) {
	// Reminder: local index will refer to the decimated mesh
	cuda::atomic_add<CURRENT_DEV>(importances[vertexIndex].viewImportance, importance);
	cuda::atomic_add<CURRENT_DEV>(sums.shadowSilhouetteImportance, importance);
}

inline CUDA_FUNCTION void record_shadow(DeviceImportanceSums<CURRENT_DEV>& sums, const float irradiance) {
	cuda::atomic_add<CURRENT_DEV>(sums.shadowImportance, irradiance);
}

inline CUDA_FUNCTION void record_direct_hit(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
									 const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint,
									 const float cosAngle, const float sharpness) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		const i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(cosAngle))
		cuda::atomic_add<CURRENT_DEV>(importances[min].viewImportance, sharpness * (1.f - ei::abs(cosAngle)));
}

inline CUDA_FUNCTION void record_direct_irradiance(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
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

inline CUDA_FUNCTION void record_indirect_irradiance(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
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

inline CUDA_FUNCTION float weight_shadow(const float importance, const SilhouetteParameters& params) {
	switch(params.shadowSizeWeight) {
		case PShadowSizeWeight::Values::INVERSE_SQR:
			return ei::sq(1.f / (1.f + importance));
		case PShadowSizeWeight::Values::INVERSE_EXP:
			return ei::exp(1.f / (1.f + importance));
		case PShadowSizeWeight::Values::INVERSE:
		default:
			return 1.f / (1.f + importance);
	}
}

inline CUDA_FUNCTION bool trace_shadow(const scene::SceneDescriptor<CURRENT_DEV>& scene, DeviceImportanceSums<CURRENT_DEV>* sums,
								const ei::Ray& shadowRay, SilPathVertex& vertex, const float importance,
								const scene::PrimitiveHandle& shadowHitId, const float lightDistance,
								const float firstShadowDistance, const LightType lightType,
								const u32 lightOffset, const ei::Vec3& otherNeeRad, const SilhouetteParameters& params) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	// TODO: what about non-manifold meshes?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(scene, backfaceRay, vertex.get_geometric_normal(),
																   lightDistance - firstShadowDistance + DIST_EPSILON);
	if(secondHit.hitId.instanceId < 0 || secondHit.hitId == vertex.get_primitive_id())
		return false;

	ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (firstShadowDistance + secondHit.distance), shadowRay.direction };
	const auto thirdHit = scene::accel_struct::first_intersection(scene, silhouetteRay, secondHit.normal,
																  lightDistance - firstShadowDistance - secondHit.distance + DIST_EPSILON);
	//if(thirdHit.hitId == vertex.get_primitive_id()) {
		// Compute the (estimated) size of the shadow region
		const ei::Plane neePlane{ shadowRay.direction, shadowRay.origin };
		const float shadowRegionSizeEstimate = estimate_shadow_light_size(scene, lightType, lightOffset,
																		  shadowRay, vertex, neePlane, firstShadowDistance,
																		  lightDistance - firstShadowDistance);
		// Scale the irradiance with the predicted shadow size
		const float weightedImportance = importance * weight_shadow(shadowRegionSizeEstimate, params);
		vertex.ext().neeWeightedIrradiance = weightedImportance;
		vertex.ext().shadowInstanceId = secondHit.hitId.instanceId;

		//const auto lodIdx = scene.lodIndices[shadowHitId.instanceId];
		//record_shadow(sums[lodIdx], weightedImportance);

		// Also check for silhouette interaction here
		if(secondHit.hitId.instanceId == shadowHitId.instanceId) {
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
			if(sharedVertices >= 2) {
				// Compute the (estimated) size of the shadow region
				// Originally we used the projected length of the shadow edge,
				// but it isn't well defined and inconsistent with the shadow
				// importance sum
				vertex.ext().silhouetteRegionSize = estimate_shadow_light_size(scene, lightType, lightOffset,
																			   shadowRay, vertex, neePlane,
																			   firstShadowDistance, lightDistance - firstShadowDistance);
				/*vertex.ext().silhouetteRegionSize = estimate_silhouette_light_size(scene, lightType, lightOffset,
																				   shadowRay, shadowHitId, vertex,
																				   neePlane, firstShadowDistance,
																				   lightDistance - firstShadowDistance);*/
			}
		}

		return true;
	//}
}

} // namespace

inline CUDA_FUNCTION void sample_importance(pt::SilhouetteTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
											const scene::SceneDescriptor<CURRENT_DEV>& scene,
											const SilhouetteParameters& params,
											const Pixel& coord, math::Rng& rng,
											Importances<CURRENT_DEV>** importances,
											DeviceImportanceSums<CURRENT_DEV>* sums) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	// We gotta keep track of our vertices
	// TODO: flexible length!
#ifdef __CUDA_ARCH__
	SilPathVertex vertices[16u];
#else // __CUDA_ARCH__
	static thread_local SilPathVertex vertices[16u];
#endif // __CUDA_ARCH__
	if(params.maxPathLength >= sizeof(vertices) / sizeof(*vertices))
		mAssertMsg(false, "Path length out of bounds!");
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

				// TODO: any_intersection-like method with both normals please...
				const auto shadowHit = scene::accel_struct::first_intersection(scene, shadowRay,
																			   nee.geoNormal,
																			   nee.dist - 0.000125f);
				const float firstShadowDistance = shadowHit.distance;
				AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

				const float weightedRadianceLuminance = get_luminance(throughput * mis * radiance) * (1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				const float weightedIrradianceLuminance = get_luminance(throughput * irradiance) *(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				if(shadowHit.hitId.instanceId < 0) {
					if(params.show_direct()) {
						mAssert(!isnan(mis));
						// Save the radiance for the later indirect lighting computation
						// Compute how much radiance arrives at the previous vertex from the direct illumination
						// Add the importance

						const auto& hitId = vertices[pathLen].get_primitive_id();
						const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
						const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;
						record_direct_irradiance(lod.polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId,
													numVertices, vertices[pathLen].get_position(), params.lightWeight * weightedIrradianceLuminance);
					}
				} else if(pathLen == 1) {
					//m_decimaters[scene.lodIndices[shadowHit.hitId.instanceId]]->record_shadow(get_luminance(throughput.weight * irradiance));

					// Determine the "rest of the direct" radiance
					const u64 ambientNeeSeed = rng.next();
					ei::Vec3 rad{ 0.f };
					const int neeCount = ei::max<int>(1, params.neeCount);
					const scene::Point vertexPos = vertices[pathLen].get_position();
					const scene::Point vertexNormal = vertices[pathLen].get_geometric_normal();
					for (int i = 0; i < neeCount; ++i) {
						math::RndSet2 currNeeRnd = rng.next();
						auto currNee = scene::lights::connect(scene, i, neeCount,
															  ambientNeeSeed, vertexPos,
															  currNeeRnd);
						Pixel outCoord;
						auto currValue = vertices[pathLen].evaluate(currNee.dir.direction, scene.media, outCoord);
						if (currNee.cosOut != 0) value.cosOut *= currNee.cosOut;
						mAssert(!isnan(currValue.value.x) && !isnan(currValue.value.y) && !isnan(currValue.value.z));
						const Spectrum currRadiance = currValue.value * currNee.diffIrradiance;
						if (any(greater(currRadiance, 0.0f)) && currValue.cosOut > 0.0f) {
							bool anyhit = scene::accel_struct::any_intersection(
								scene, vertexPos, currNee.position,
								vertexNormal, currNee.geoNormal,
								currNee.dir.direction);
							if (!anyhit) {
								AreaPdf currHitPdf = currValue.pdf.forw.to_area_pdf(currNee.cosOut, currNee.distSq);
								// TODO: it seems that, since we're looking at irradiance here (and also did not weight
								// the previous weightedIrradiance with 1/(neeCount + 1)) we must not use the regular
								// MIS weight here
								float curMis = 1.0f / (1 + currHitPdf / currNee.creationPdf);
								mAssert(!isnan(curMis));
								rad += currValue.cosOut * currRadiance * curMis;
							}
						}
					}

					vertices[pathLen].ext().otherNeeLuminance = get_luminance(rad);
					// TODO: use this radiance to conditionally discard importance
					trace_shadow(scene, sums, shadowRay, vertices[pathLen], weightedRadianceLuminance,
								 shadowHit.hitId, nee.dist, firstShadowDistance,
								 lightType, lightOffset, rad * throughput, params);
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

		if(params.show_view()) {
			record_direct_hit(lod.polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId, numVertices, vertices[pathLen].get_position(),
							  -ei::dot(vertices[pathLen + 1].get_incident_direction(), vertices[pathLen + 1].get_normal()),
							  params.viewWeight * sharpness);
		}
		++pathLen;
	} while(pathLen < params.maxPathLength);

	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{ 0.f };
	for(int p = pathLen; p >= 1; --p) {
		// Last vertex doesn't have indirect contribution
		if(p < pathLen) {
			accumRadiance = vertices[p].ext().throughput * accumRadiance + (vertices[p + 1].ext().shadowInstanceId == 0 ?
																			vertices[p + 1].ext().pathRadiance : ei::Vec3{ 0.f });
			const ei::Vec3 irradiance = vertices[p].ext().outCos * accumRadiance;

			const auto& hitId = vertices[p].get_primitive_id();
			const auto* lod = &scene.lods[scene.lodIndices[hitId.instanceId]];
			const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;

			const float importance = get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
			if(params.show_indirect()) {
				record_indirect_irradiance(lod->polygon, importances[scene.lodIndices[hitId.instanceId]], hitId.primId,
										   numVertices, vertices[p].get_position(), params.lightWeight * importance);
			}
		}
		// TODO: store accumulated sharpness
		// Check if it is sensible to keep shadow silhouettes intact
		// TODO: replace threshold with something sensible
		const auto& ext = vertices[p].ext();


		if(p == 1 && ext.shadowInstanceId >= 0) {
			// TODO: factor in background illumination too
			const float indirectLuminance = get_luminance(accumRadiance) + ext.otherNeeLuminance;
			const float totalLuminance = get_luminance(ext.pathRadiance) + indirectLuminance;
			const float ratio = totalLuminance / indirectLuminance - 1.f;
			if(ratio > params.directIndirectRatio && params.show_silhouette()) {
				// Regular shadow importance
				const auto lodIdx = scene.lodIndices[ext.shadowInstanceId];
				record_shadow(sums[lodIdx], ext.neeWeightedIrradiance);

				if(ext.silhouetteRegionSize >= 0.f) {
					constexpr float FACTOR = 2'000.f;

					// Idea: we have one NEE for silhouette stuff and n other ones to estimate the
					// brightness; all of them contribute to the direct irradiance thingy,
					// but only one acts as a silhouette detector(?)
					// Kinda sucks though

					/*trace_shadow(scene, sums, shadowRay, vertices[pathLen], weightedIrradianceLuminance,
						shadowHit.hitId, nee.dist, firstShadowDistance,
						lightType, lightOffset, params);*/

						// TODO: proper factor!
					const float silhouetteImportance = weight_shadow(ext.silhouetteRegionSize, params)
						* params.shadowSilhouetteWeight * FACTOR * (totalLuminance - indirectLuminance);

					for(i32 i = 0; i < 2; ++i) {
						mAssert(ext.silhouetteVerticesFirst[i] >= 0 && static_cast<u32>(ext.silhouetteVerticesFirst[i]) < scene.lods[lodIdx].polygon.numVertices);
						record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx],
															  ext.silhouetteVerticesFirst[i],
															  silhouetteImportance);
						if(ext.silhouetteVerticesSecond[i] >= 0 && ext.silhouetteVerticesFirst[i] != ext.silhouetteVerticesSecond[i]) {
							mAssert(static_cast<u32>(ext.silhouetteVerticesSecond[i]) < scene.lods[lodIdx].polygon.numVertices);
							record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx],
																  ext.silhouetteVerticesSecond[i],
																  silhouetteImportance);
						}
					}
				}
				outputBuffer.template contribute<PShadowRecorded>(coord, 1.f);
			} else {
				outputBuffer.template contribute<PShadowOmitted>(coord, 1.f);
			}
		}
		
	}
}

inline CUDA_FUNCTION void sample_vis_importance(pt::SilhouetteTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
										 const scene::SceneDescriptor<CURRENT_DEV>& scene,
										 const Pixel& coord, math::Rng& rng,
										 Importances<CURRENT_DEV>** importances,
										 DeviceImportanceSums<CURRENT_DEV>* sums,
										 const float maxImportance) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	float guideWeight = 1.0f;
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) == WalkResult::HIT) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		const auto& lod = scene.lods[lodIdx];
		const auto& polygon = lod.polygon;
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

		outputBuffer.contribute<ImportanceTarget>(coord, importance / maxImportance);
		outputBuffer.contribute<PolyShareTarget>(coord, sums[lodIdx].shadowImportance / static_cast<float>(lod.numPrimitives));
	}
}

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt