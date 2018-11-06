#pragma once

#include "accel_struct.hpp"
#include "instance.hpp"
#include "export/dll_export.hpp"
#include "handles.hpp"
#include "core/cameras/camera.hpp"
#include "ei/3dtypes.hpp"
#include <memory>
#include <vector>

namespace mufflon::scene {

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class LIBRARY_API Scene {
public:
	Scene() = default;
	Scene(const Scene&) = default;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = default;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Add an instance to be rendered
	void add_instance(InstanceHandle hdl) {
		m_instances.push_back(hdl);
		m_boundingBox = ei::Box(m_boundingBox, hdl->get_bounding_box());
		m_accelDirty = true;
	}

	// Synchronizes entire scene to the device
	template < Device dev >
	void synchronize() {
		// Create camera parameters on CPU side (required for other devices as copy source)
		m_cameraParams.get<unique_device_ptr<Device::CPU, char>>() =
			make_udevptr_array<Device::CPU, char>(m_camera->get_parameter_pack_size());
		// [Weird] using the following two lines as a one-liner causes an internal compiler bug.
		// m_camera->get_parameter_pack(as<cameras::CameraParams>(*m_cameraParams.get<unique_device_ptr<Device::CPU, char>>())); // This line causes the bug
		auto& pCpu = m_cameraParams.get<unique_device_ptr<Device::CPU, char>>();
		m_camera->get_parameter_pack(as<cameras::CameraParams>(*pCpu));
		// Copy to the real target device, if it differs from the CPU
		if constexpr(dev == Device::CUDA) {
			m_cameraParams.get<unique_device_ptr<Device::CUDA, char>>() =
				make_udevptr_array<Device::CUDA, char>(m_camera->get_parameter_pack_size());
			// [Weird] Again, using the RHS from the next line directly without storing it
			// in a variable causes very mysterious bugs.
			auto* pTarget = m_cameraParams.get<unique_device_ptr<Device::CUDA, char>>().get();
			cudaMemcpy(pTarget, pCpu.get(), m_camera->get_parameter_pack_size(), cudaMemcpyHostToDevice);
		}

		for(InstanceHandle instance : m_instances) {
			// TODO
		}
		// TODO: materials etc.
	}

	template < Device dev >
	void unload() {
		m_cameraParams.get<unique_device_ptr<dev, cameras::CameraParams>>().release();
		for(InstanceHandle instance : m_instances) {
			// TODO
		}
		// TODO: materials etc.
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	// Checks if the acceleration structure on one of the system parts has been modified.
	bool is_accel_dirty(Device res) const noexcept;

	// Checks whether the object currently has a BVH.
	bool has_accel_structure() const noexcept {
		return m_accel_struct != nullptr;
	}
	// Returns the BVH of this object.
	const IAccelerationStructure& get_accel_structure() const noexcept {
		mAssert(this->has_accel_structure());
		return *m_accel_struct;
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

	// Overwrite which camera is used of the scene
	void set_camera(ConstCameraHandle camera) noexcept {
		mAssert(camera != nullptr);
		m_camera = camera;
	}
	// Access the active camera
	ConstCameraHandle get_camera() const noexcept {
		return m_camera;
	}

private:
	// List of instances and thus objects to-be-rendered
	std::vector<InstanceHandle> m_instances;

	ConstCameraHandle m_camera;	// The single, chosen camera for rendering this scene
	util::TaggedTuple<unique_device_ptr<Device::CPU, char>, // TODO: alias/wrapper type for a TaggedTuple of an identic resource for all devices?
		unique_device_ptr<Device::CUDA, char>> m_cameraParams;

	// TODO: cameras, lights, materials
	// Acceleration structure over all instances
	bool m_accelDirty = false;
	std::unique_ptr<IAccelerationStructure> m_accel_struct = nullptr;

	ei::Box m_boundingBox;
};

} // namespace mufflon::scene