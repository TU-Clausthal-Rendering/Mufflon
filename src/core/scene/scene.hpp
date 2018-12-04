#pragma once

#include "instance.hpp"
#include "handles.hpp"
#include "types.hpp"
#include "core/scene/accel_structs/accel_struct.hpp"
#include "lights/light_tree.hpp"
#include "core/cameras/camera.hpp"
#include "core/memory/generic_resource.hpp"
#include <memory>
#include <vector>

namespace mufflon { namespace scene {

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class Scene {
public:
	Scene(ConstCameraHandle cam) :
		m_camera(cam) {}
	Scene(const Scene&) = delete;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = delete;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Add an instance to be rendered
	void add_instance(InstanceHandle hdl) {
		m_instances.push_back(hdl);
		m_boundingBox = ei::Box(m_boundingBox, hdl->get_bounding_box());
		m_accelDirty = true;
	}

	void load_media(const std::vector<materials::Medium>& media);

	// Synchronizes entire scene to the device
	template < Device dev >
	void synchronize() {
		for(InstanceHandle instance : m_instances) {
			(void)instance;
			// TODO
		}
		// TODO: materials etc.
		m_media.synchronize<dev, Device::CPU>();
	}

	template < Device dev >
	void unload() {
		for(InstanceHandle instance : m_instances) {
			(void)instance;
			// TODO
		}
		// TODO: materials etc.
		m_media.unload<dev>();
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	// Checks if the acceleration structure on one of the system parts has been modified.
	bool is_accel_dirty(Device res) const noexcept;

	// Checks whether the object currently has a BVH.
	bool has_accel_structure() const noexcept {
		return m_accelStruct != nullptr;
	}
	// Returns the BVH of this object.
	const accel_struct::IAccelerationStructure& get_accel_structure() const noexcept {
		mAssert(this->has_accel_structure());
		return *m_accelStruct;
	}
	// Clears the BVH of this object.
	void clear_accel_structure();
	// Initializes the acceleration structure to a given implementation.
	template < class Accel, class... Args >
	void set_accel_structure(Args&& ...args) {
		m_accel_struct = std::make_unique<Accel>(std::forward<Args>(args)...);
	}

	// (Re-)builds the acceleration structure
	void build_accel_structure();

	void set_lights(std::vector<lights::PositionalLights>&& posLights,
					std::vector<lights::DirectionalLight>&& dirLights,
					TextureHandle envLightTexture = nullptr) {
		m_lightTree.build(std::move(posLights), std::move(dirLights),
						  m_boundingBox, envLightTexture);
	}

	template < Device dev >
	const lights::LightTree<dev>& get_light_tree() {
		return m_lightTree.aquire_tree<dev>();
	}

	// Overwrite which camera is used of the scene
	void set_camera(ConstCameraHandle camera) noexcept {
		mAssert(camera != nullptr);
		m_camera = camera;
	}
	// Access the active camera
	ConstCameraHandle get_camera() const noexcept {
		return m_camera;
	}

	template < Device dev >
	const materials::Medium* get_media() const noexcept {
		return m_media.get<dev, materials::Medium>();
	}

private:
	// List of instances and thus objects to-be-rendered
	std::vector<InstanceHandle> m_instances;
	GenericResource m_media;		// Device copy of the media. It is not possible to access the world from a CUDA compiled file.
	ConstCameraHandle m_camera;		// The single, chosen camera for rendering this scene

	// Light tree containing all light sources enabled for the scene
	lights::LightTreeBuilder m_lightTree;
	// TODO: materials
	// Acceleration structure over all instances
	bool m_accelDirty = false;
	std::unique_ptr<accel_struct::IAccelerationStructure> m_accelStruct = nullptr;

	ei::Box m_boundingBox;
};

}} // namespace mufflon::scene