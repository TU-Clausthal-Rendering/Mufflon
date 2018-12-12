#pragma once

#include <cstdint>

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
	struct HandlePack;
}
namespace lights {
	struct PointLight;
	struct SpotLight;
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
using PrimitiveHandle = uint64_t;		// | 32bit instance | 32 bit primitive (poly+spheres enumerated in a single sequence |

}} // namespace mufflon::scene