#pragma once

#include "light_tree.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/materials/point_medium.hpp"

namespace mufflon { namespace scene { namespace lights {

// Sets the medium of a given light by tracing a ray in the scene and determining
// the medium of the hitpoint
inline CUDA_FUNCTION void set_light_medium(char* mem, LightType type, const SceneDescriptor<CURRENT_DEV>& scene) {
	switch(type) {
		case LightType::POINT_LIGHT:
			reinterpret_cast<PointLight*>(mem)->mediumIndex = materials::get_point_medium(scene, reinterpret_cast<PointLight*>(mem)->position);
			break;
		case LightType::SPOT_LIGHT:
			reinterpret_cast<SpotLight*>(mem)->mediumIndex = materials::get_point_medium(scene, reinterpret_cast<SpotLight*>(mem)->position);
			break;
		default: return;
	}
}

}}} // namespace mufflon::scene::lights