#pragma once

#include "util/assert.hpp"
#include "core/export/core_api.h"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/lights.hpp"
#include <ei/vector.hpp>
#include <ei/3dtypes.hpp>
#include <ei/3dintersection.hpp>
#include <cuda_runtime.h>

namespace mufflon { namespace renderer { namespace decimaters { namespace combined {



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

inline CUDA_FUNCTION ei::Triangle get_world_light_triangle(const scene::SceneDescriptor<CURRENT_DEV>& scene,
														   const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const scene::lights::AreaLightTriangle<CURRENT_DEV>*>(lightMemory);
	// Returns v0, v1, v2
	return ei::Triangle{
		light.posV[0], light.posV[1] + light.posV[0], light.posV[2] + light.posV[0]
	};
}

inline CUDA_FUNCTION ei::Tetrahedron get_world_light_quad(const scene::SceneDescriptor<CURRENT_DEV>& scene,
														  const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const scene::lights::AreaLightQuad<CURRENT_DEV>*>(lightMemory);
	// Return v0, v1, v2, v3
	const ei::Vec3 v3 = light.posV[1] + light.posV[0];
	return ei::Tetrahedron{
		light.posV[0], light.posV[2] + light.posV[0], light.posV[3] + light.posV[2] + v3, v3
	};
}

inline CUDA_FUNCTION ei::Triangle get_world_light_triangle_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																 const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const scene::lights::AreaLightTriangle<CURRENT_DEV>*>(lightMemory);
	// Returns v1 - v0, v2 - v0, v1 - v2
	return ei::Triangle{ light.posV[1], light.posV[2],
						 light.posV[1] - light.posV[2] };
}

inline CUDA_FUNCTION ei::Tetrahedron get_world_light_quad_sides(const scene::SceneDescriptor<CURRENT_DEV>& scene,
																const u32 offset) {
	const auto* lightMemory = scene.lightTree.posLights.memory + offset;
	const auto& light = *reinterpret_cast<const scene::lights::AreaLightQuad<CURRENT_DEV>*>(lightMemory);
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

inline CUDA_FUNCTION float estimate_shadow_light_size(const scene::SceneDescriptor<CURRENT_DEV>& scene,
													  const scene::lights::LightType lightType, const u32 lightOffset,
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
			const auto& light = *reinterpret_cast<const scene::lights::AreaLightSphere<CURRENT_DEV>*>(scene.lightTree.posLights.memory + lightOffset);
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

}}}} // namespace mufflon::renderer::decimaters::combined