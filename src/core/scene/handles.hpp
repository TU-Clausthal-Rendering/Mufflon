#pragma once

namespace mufflon { namespace cameras {
class Camera;
}}// namespace mufflon::cameras

namespace mufflon { namespace scene {

class Object;
class Instance;
class Scene;
namespace materials {
class IMaterial;
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
using ConstTextureHandle = const textures::Texture*;
using TextureHandle = textures::Texture*;

}} // namespace mufflon::scene