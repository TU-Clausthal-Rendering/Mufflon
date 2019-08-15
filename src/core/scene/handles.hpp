#pragma once

#include "util/int_types.hpp"

namespace mufflon { namespace cameras {
class Camera;
struct CameraParams;
}}// namespace mufflon::cameras

namespace mufflon { namespace scene {

class Object;
class Instance;
class Scene;
class Scenario;
namespace materials {
	class IMaterial;
	class Medium;
	using MediumHandle = uint16_t;
	struct MaterialDescriptorBase;
}
namespace lights {
	struct PointLight;
	struct SpotLight;
	struct DirectionalLight;
	class Background;
}
namespace textures {
	class Texture;
}


using ObjectHandle = Object*;
using ConstObjectHandle = const Object*;
using InstanceHandle = Instance*;
using ConstInstanceHandle = const Instance*;
using MaterialHandle = materials::IMaterial*;
using ConstMaterialHandle = const materials::IMaterial*;
using CameraHandle = cameras::Camera*;
using ConstCameraHandle = const cameras::Camera*;
using SceneHandle = Scene*;
using ConstSceneHandle = const Scene*;
using ScenarioHandle = Scenario*;
using ConstScenarioHandle = const Scenario*;
using ConstTextureHandle = const textures::Texture*;
using TextureHandle = textures::Texture*;
struct PrimitiveHandle {
	i32 instanceId { -1 };
	i32 primId { -1 };
	constexpr bool operator==(const PrimitiveHandle& rhs) const noexcept {
		return instanceId == rhs.instanceId && primId == rhs.primId;
	}
	constexpr bool operator!=(const PrimitiveHandle& rhs) const noexcept {
		return instanceId != rhs.instanceId || primId != rhs.primId;
	}
	constexpr bool is_valid() const noexcept {
		return instanceId != -1 && primId != -1;
	}
};

}} // namespace mufflon::scene
