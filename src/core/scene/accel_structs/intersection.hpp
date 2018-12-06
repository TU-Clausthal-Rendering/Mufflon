#pragma once

#include "accel_structs_commons.hpp"
#include "core/scene/handles.hpp"
#include <ei/3dtypes.hpp>

namespace mufflon {
namespace scene {
namespace accel_struct {

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

void first_intersection_CUDA(
	AccelStructInfo bvh,
	RayInfo rayInfo,
	RayIntersectionResult* result);

bool any_intersection_CUDA(
	AccelStructInfo bvh,
	RayInfo rayInfo);

}
}
}