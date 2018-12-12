#pragma once

#include "util/types.hpp"
#include "util/assert.hpp"
#include "core/export/api.h"

namespace mufflon {
namespace scene {
namespace accel_struct {

CUDA_FUNCTION constexpr void extract_prim_counts(i32 primitiveCount, ei::IVec4& count) {
	i32 sphCountMask = 0x000003FF;
	i32 triCountMask = 0x3FF00000;
	i32 quadCountMask = 0x000FFC00;
	i32 triShift = 20;
	i32 quadShift = 10;
	count.x = (primitiveCount & triCountMask) >> triShift;
	count.y = (primitiveCount & quadCountMask) >> quadShift;
	count.z = (primitiveCount & sphCountMask);
	count.w = count.x + count.y + count.z;
}

struct AccelStructInfo {
	struct Size
	{
		i32 offsetSpheres;
		i32 offsetQuads;
		i32 numVertices;
		i32 numPrimives;
		i32 bvhSize; // Number of ei::Vec4 in bvh.
	} sizes;
	struct InputArrays
	{
		ei::Vec3* meshVertices;
		ei::Vec2* meshUVs;
		i32* triIndices;
		i32* quadIndices;
		ei::Vec4* spheres;
	} inputs;
	struct OutputArrays {
		ei::Vec4* bvh;
		i32* primIds;
	} outputs;
};

CUDA_FUNCTION float int_as_float(i32 v) {
#ifdef __CUDA_ARCH__
	return __int_as_float(v);
#else
	return reinterpret_cast<float&>(v);
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION i32 float_as_int(float v) {
#ifdef __CUDA_ARCH__
	return __float_as_int(v);
#else
	return reinterpret_cast<i32&>(v);
#endif // __CUDA_ARCH__
}

}
}
}