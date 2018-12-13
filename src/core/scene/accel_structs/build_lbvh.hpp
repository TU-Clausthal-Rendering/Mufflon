#pragma once

#include "util/types.hpp"
#include "intersection.hpp"
#include "accel_structs_commons.hpp"

namespace mufflon { namespace scene { namespace accel_struct {

 template < Device dev >
 void build_lbvh_obj(ObjectDescriptor<dev>& obj, const ei::Box& aabb);

 template < Device dev >
 void build_lbvh_scene(SceneDescriptor<dev>& scene);

}}} // namespace mufflon::scene::accel_struct