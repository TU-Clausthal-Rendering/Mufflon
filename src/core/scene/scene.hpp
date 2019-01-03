#pragma once

#include "instance.hpp"
#include "handles.hpp"
#include "types.hpp"
#include "lights/light_tree.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/scene/accel_structs/accel_struct_info.hpp"
#include "core/scene/object.hpp"
#include "core/scene/accel_structs/lbvh.hpp"
#include <memory>
#include <tuple>
#include <vector>

namespace mufflon { namespace scene {

template < Device dev >
struct SceneDescriptor;

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class Scene {
public:
	Scene(ConstCameraHandle cam,
		  const std::vector<std::unique_ptr<materials::IMaterial>>& materialsRef) :
		m_camera(cam),
		m_materialsRef(materialsRef)
	{}
	Scene(const Scene&) = delete;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = delete;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Add an instance to be rendered
	void add_instance(InstanceHandle hdl) {
		m_instances.push_back(hdl);
		m_boundingBox = ei::Box(m_boundingBox, hdl->get_bounding_box());
		clear_accel_structure();
	}

	void load_media(const std::vector<materials::Medium>& media);

	// Synchronizes entire scene to the device
	template < Device dev >
	void synchronize() {
		for(InstanceHandle instance : m_instances) {
			(void)instance;
			// TODO
		}
		m_lightTree.synchronize<dev>();
		m_media.synchronize<dev>();
	}

	template < Device dev >
	void unload() {
		for(InstanceHandle instance : m_instances) {
			(void)instance;
			// TODO
		}
		// TODO: materials etc.
		m_lightTree.unload<dev>();
		m_media.unload<dev>();
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	// Checks if the acceleration structure on one of the system parts has been modified.
	template < Device dev >
	bool is_accel_dirty() const noexcept {
		return m_accelStruct[get_device_index<dev>()].type == accel_struct::AccelType::NONE;
	}

	// Checks whether the scene currently has a BVH.
	/*bool has_accel_structure() const noexcept {
		return m_accelStruct.type != accel_struct::Type::NONE;
	}*/
	// Clears the BVH of this object.
	void clear_accel_structure() {
		m_accelStruct.mark_invalid();
	}

	void set_lights(std::vector<lights::PositionalLights>&& posLights,
					std::vector<lights::DirectionalLight>&& dirLights,
					TextureHandle envLightTexture = nullptr) {
		m_lightTree.build(std::move(posLights), std::move(dirLights),
						  m_boundingBox, envLightTexture);
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


	/**
	 * Creates a single structure which grants access to all scene data
	 * needed on the specified device.
	 * Synchronizes implicitly.
	 * It needs to be passed up to three tuples, each coupling names and types for
	 * vertex, face, and sphere attributes which the renderer wants to have
	 * access to.
	 *
	 * Usage example:
	 * scene::SceneDescriptor<Device::CUDA> sceneDesc = m_currentScene->get_descriptor<Device::CUDA>(
	 *		std::make_tuple(scene::geometry::Polygons::VAttrDesc<int>{"T1"},
	 *						scene::geometry::Polygons::VAttrDesc<int>{"T2"}),
	 *		{}, // No face attributes
	 *		std::make_tuple(scene::geometry::Spheres::AttrDesc<float>{"S1"})
	 * );
	 */
	template < Device dev >
	SceneDescriptor<dev> get_descriptor(const std::vector<const char*>& vertexAttribs,
										const std::vector<const char*>& faceAttribs,
										const std::vector<const char*>& sphereAttribs,
										const ei::IVec2& resolution);

	template < Device dev >
	void update_camera_medium(SceneDescriptor<dev>& scene);

private:
	// List of instances and thus objects to-be-rendered
	std::vector<InstanceHandle> m_instances;
	GenericResource m_media;		// Device copy of the media. It is not possible to access the world from a CUDA compiled file.
	ConstCameraHandle m_camera;		// The single, chosen camera for rendering this scene
	const std::vector<std::unique_ptr<materials::IMaterial>>& m_materialsRef;	// Refer the world's materials. There is no scenario dependent filtering because of global indexing.
	GenericResource m_materials;	// Device instanciation of Material parameter packs and an offset table (first table then data).

	// Light tree containing all light sources enabled for the scene
	lights::LightTreeBuilder m_lightTree;

	// Acceleration structure over all instances
	accel_struct::LBVHBuilder m_accelStruct;

	// Resources for descriptors
	util::TaggedTuple<unique_device_ptr<Device::CPU, ObjectDescriptor<Device::CPU>[]>,
		unique_device_ptr<Device::CUDA, ObjectDescriptor<Device::CUDA>[]>> m_objDevDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, ei::Mat3x4[]>,
		unique_device_ptr<Device::CUDA, ei::Mat3x4[]>> m_instTransformsDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, u32[]>,
		unique_device_ptr<Device::CUDA, u32[]>> m_instObjIndicesDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, ei::Box[]>,
		unique_device_ptr<Device::CUDA, ei::Box[]>> m_objAabbsDesc;

	ei::Box m_boundingBox;


	template< Device dev >
	void load_materials();
	
	void update_camera_medium_cpu(SceneDescriptor<Device::CPU>& scene);
};

namespace scene_detail {

void update_camera_medium_cuda(SceneDescriptor<Device::CUDA>& scene);

} // namespace scene_detail

}} // namespace mufflon::scene