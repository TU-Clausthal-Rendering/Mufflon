#pragma once

#include "util/types.hpp"
#include "intersection.hpp"
#include "accel_structs_commons.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

void build_lbvh64_info(AccelStructInfo::Size& sizes,
	AccelStructInfo::InputArrays& inputs, AccelStructInfo::OutputArrays& ouputs,
	ei::Box& bbox, ei::Vec4& traverseCost);

 void build_lbvh64(ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec3 lo, ei::Vec3 hi, ei::Vec4 traverseCosts, i32 numPrimitives,
	i32 offsetQuads, i32 offsetSpheres, i32** primIds, ei::Vec4** bvh, i32& bvhSize);

}}} // namespace mufflon::scene::accel_struct