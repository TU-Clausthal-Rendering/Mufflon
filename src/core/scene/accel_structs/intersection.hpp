#pragma once

#include "accel_structs_commons.hpp"
#include "core/scene/handles.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"

#include <ei/3dtypes.hpp>

// TODO: use u32 for primiIds.

namespace mufflon {
namespace scene {
namespace accel_struct {

// The surface coordinates of the hitpoint, depending on primitive type
union SurfaceParametrization {
	ei::Vec2 barycentric;	// Triangle
	ei::Vec2 bilinear;		// Quad
	ei::Vec2 st;			// Sphere

	__host__ __device__ SurfaceParametrization() {}
};

struct RayIntersectionResult {
	float hitT;
	struct HitID {
		i32 instanceId;
		i32 primId;

		__host__ __device__ u32 get_primitive_id() const {
			return static_cast<u32>(primId & 0x7FFFFFFF); // Remove the bit for identifying quad sides
		}
	} hitId;
	ei::Vec3 normal;
	ei::Vec3 tangentX;
	ei::Vec3 tangentY;
	ei::Vec2 uv;
	SurfaceParametrization surfaceParams;
};

struct RayInfo {
	ei::Ray ray;
	i32 startPrimId;
	float tmin;
	float tmax;
};

template < Device dev >
__host__ __device__
bool any_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

template < Device dev >
__host__ __device__
RayIntersectionResult first_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

}
}
}