#pragma once

#include "util/types.hpp"

namespace mufflon {

ei::Vec4* build_lbvh64(ei::Vec3* triVertices,
	ei::Vec3* quadVertices,
	ei::Vec4* sphVertices,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec3 lo, ei::Vec3 hi, ei::Vec4 traverseCosts,
	i32 numTriangles, i32 numQuads, i32 numSpheres);

}

