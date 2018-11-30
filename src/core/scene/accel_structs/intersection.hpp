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

struct RayInfo {
	ei::Ray ray;
	i32 startPrimId;
	float tmin;
	float tmax;
};

struct AccelStructInfo {
	ei::Vec4* bvh;
	i32 bvhSize;
	ei::Vec3* meshVertices;
	ei::Vec2* meshUVs;
	i32* triIndices;
	i32* quadIndices;
	ei::Vec4* spheres;
	i32 offsetQuads; 
	i32 offsetSpheres;
	i32* primIds;
	i32 numPrimives;
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