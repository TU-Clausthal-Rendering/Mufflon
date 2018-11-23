#pragma once

#include "util/types.hpp"
#include <ei/3dtypes.hpp>

namespace mufflon {
namespace scene {
namespace accel_struct {

struct RayIntersectionResult {
	float hitT;
	i32 hitPrimId;
	ei::Vec3 normal;
	ei::Vec3 tangent;
	ei::Vec2 uv;
	ei::Vec3 barycentric;
};

void intersection_test_CUDA(const ei::Ray ray, const i32 startPrimId,
	const ei::Vec4* bvh,
	const ei::Vec3* triVertices,
	const ei::Vec3* quadVertices,
	const ei::Vec4* sphVertices,
	const ei::Vec2* triUVs,
	const ei::Vec2* quadUVs,
	const i32* triIndices,
	const i32* quadIndices,
	const i32* primIds,
	const i32 offsetQuads, const i32 offsetSpheres,
	RayIntersectionResult* result,
	const float tmin, const float tmax);

}
}
}