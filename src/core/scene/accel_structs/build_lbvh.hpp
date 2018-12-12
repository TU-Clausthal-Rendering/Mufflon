#pragma once

#include "util/types.hpp"
#include "intersection.hpp"
#include "accel_structs_commons.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

// should not be used.
void build_lbvh64_info(AccelStructInfo::Size& sizes,
	AccelStructInfo::InputArrays& inputs, AccelStructInfo::OutputArrays& ouputs,
	ei::Box& bbox, ei::Vec4& traverseCost);

// should not be used.
 void build_lbvh64(ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec3 lo, ei::Vec3 hi, ei::Vec4 traverseCosts, i32 numPrimitives,
	i32 offsetQuads, i32 offsetSpheres, i32** primIds, ei::Vec4** bvh, i32& bvhSize);

 void build_lbvh_obj(ObjectDescriptor<Device::CPU> obj);

 void build_lbvh_obj(ObjectDescriptor<Device::CUDA> obj);
 
 void build_lbvh_scene(SceneDescriptor<Device::CPU> scene);

 void build_lbvh_scene(SceneDescriptor<Device::CUDA> scene);

}}} // namespace mufflon::scene::accel_struct