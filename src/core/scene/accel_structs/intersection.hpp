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

template < Device dev >
struct RayIntersectionResult {
	float hitT;
	PrimitiveHandle hitPrimId;
	ei::Vec3 normal;
	ei::Vec3 tangent;
	ei::Vec2 uv;
	ei::Vec3 barycentric;
};

struct RayInfo {
	ei::Ray ray;
	i32 startPrimId;
	float tmin;
	float tmax;
};

bool any_intersection_lbvh(
	SceneDescriptor<Device::CUDA> scene,
	const ei::Ray ray, const u64 startInsPrimId,
	const float tmin, const float tmax
);

bool any_intersection_lbvh(
	SceneDescriptor<Device::CPU> scene,
	const ei::Ray ray, const u64 startInsPrimId,
	const float tmin, const float tmax
);

void first_intersection_lbvh(
	SceneDescriptor<Device::CPU> scene,
	const ei::Ray ray, const u64 startInsPrimId,
	const float tmin, const float tmax,
	RayIntersectionResult<Device::CPU>* result
);

void first_intersection_lbvh(
	SceneDescriptor<Device::CUDA> scene,
	const ei::Ray ray, const u64 startInsPrimId,
	const float tmin, const float tmax,
	RayIntersectionResult<Device::CUDA>* result
);

}
}
}