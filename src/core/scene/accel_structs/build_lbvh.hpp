#pragma once

#include "util/types.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

ei::Vec4* build_lbvh64(ei::Vec3* triVertices,
	ei::Vec3* quadVertices,
	ei::Vec4* sphVertices,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec3 lo, ei::Vec3 hi, ei::Vec4 traverseCosts,
	i32 numTriangles, i32 numQuads, i32 numSpheres,
	i32** primIds, i32& offsetQuads, i32& offsetSpheres, i32& bvhSize);

}}} // namespace mufflon::scene::accel_struct