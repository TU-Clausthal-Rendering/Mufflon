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
	ei::Vec3 tangent;
	ei::Vec2 uv;
	ei::Vec3 barycentric; // TODO: storing 2 is sufficient, TODO: different coordinates for spheres/quads
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