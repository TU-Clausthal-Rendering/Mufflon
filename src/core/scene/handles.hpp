#pragma once

namespace mufflon::cameras {
	class Camera;
} // namespace mufflon::cameras

namespace mufflon::scene {

class Object;
class Instance;
class Scene;
namespace materials {
	class IMaterial;
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

} // namespace mufflon::scene