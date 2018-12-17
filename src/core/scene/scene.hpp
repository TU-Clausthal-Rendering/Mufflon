#pragma once

#include "descriptors.hpp"
#include "instance.hpp"
#include "handles.hpp"
#include "types.hpp"
#include "lights/light_tree.hpp"
#include "core/cameras/camera.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/scene/accel_structs/accel_struct_info.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/object.hpp"
#include "core/scene/accel_structs/lbvh.hpp"
#include <memory>
#include <tuple>
#include <vector>

namespace mufflon { namespace scene {

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
	template < Device dev, std::size_t N, std::size_t M, std::size_t O >
	SceneDescriptor<dev> get_descriptor(const std::array<const char*, N> &vertexAttribs,
										const std::array<const char*, M> &faceAttribs,
										const std::array<const char*, O> &sphereAttribs,
										const ei::IVec2& resolution) {
		synchronize<dev>();
		std::vector<ObjectDescriptor<dev>> objectDescs;
		std::vector<ei::Mat3x4> instanceTransformations;
		std::vector<u32> objectIndices;
		std::vector<ei::Box> objectAabbs;
		// We need this to ensure we only create one descriptor per object
		std::unordered_map<Object*, u32> objectDescMap;

		// Create the object and instance descriptors hand-in-hand
		for(InstanceHandle inst : m_instances) {
			// Create the object descriptor, if not already present, and its index
			Object* objHdl = &inst->get_object();
			auto entry = objectDescMap.find(objHdl);
			if(entry == objectDescMap.end()) {
				entry = objectDescMap.emplace(objHdl, static_cast<u32>(objectDescs.size())).first;
				objectDescs.push_back(objHdl->get_descriptor<dev>(vertexAttribs, faceAttribs, sphereAttribs));
				objectAabbs.push_back(objHdl->get_bounding_box());
			}
			instanceTransformations.push_back(inst->get_transformation_matrix());
			objectIndices.push_back(entry->second);
		}
		// Allocate the device memory and copy over the descriptors
		auto& objDevDesc = m_objDevDesc.get<unique_device_ptr<dev, ObjectDescriptor<dev>>>();
		objDevDesc = make_udevptr_array<dev, ObjectDescriptor<dev>>(objectDescs.size());
		copy(objDevDesc.get(), objectDescs.data(), objectDescs.size() * sizeof(ObjectDescriptor<dev>));

		auto& instTransformsDesc = m_instTransformsDesc.get<unique_device_ptr<dev, ei::Mat3x4>>();
		instTransformsDesc = make_udevptr_array<dev, ei::Mat3x4>(instanceTransformations.size());
		copy(instTransformsDesc.get(), instanceTransformations.data(), sizeof(ei::Mat3x4) * instanceTransformations.size());

		auto& instObjIndicesDesc = m_instObjIndicesDesc.get<unique_device_ptr<dev, u32>>();
		instObjIndicesDesc = make_udevptr_array<dev,u32>(objectIndices.size());
		copy(instObjIndicesDesc.get(), objectIndices.data(), sizeof(u32) * objectIndices.size());

		auto& objAabbsDesc = m_objAabbsDesc.get<unique_device_ptr<dev, ei::Box>>();
		objAabbsDesc = make_udevptr_array<dev, ei::Box>(objectAabbs.size());
		copy(objAabbsDesc.get(), objectAabbs.data(), sizeof(ei::Box) * objectAabbs.size());

		load_materials<dev>();

		// Camera
		CameraDescriptor camera{};
		m_camera->get_parameter_pack(&camera.get(), resolution);

		SceneDescriptor<dev> sceneDesc{
			camera,
			static_cast<u32>(objectDescs.size()),
			static_cast<u32>(m_instances.size()),
			m_boundingBox,
			objDevDesc.get(),
			AccelDescriptor{},
			instTransformsDesc.get(),
			instObjIndicesDesc.get(),
			objAabbsDesc.get(),
			m_lightTree.acquire_const<dev>(),
			as<materials::Medium>(m_media.acquire_const<dev>()),
			as<int>(m_materials.acquire_const<dev>())
		};

		// Rebuild Instance BVH?
		
		if(m_accelStruct.needs_rebuild<dev>()) {
			//m_accelStruct.build(sceneDesc);
		}
		//sceneDesc.accelStruct = m_accelStruct.acquire_const<dev>();

		return sceneDesc;
	}


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
	util::TaggedTuple<unique_device_ptr<Device::CPU, ObjectDescriptor<Device::CPU>>,
		unique_device_ptr<Device::CUDA, ObjectDescriptor<Device::CUDA>>> m_objDevDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, ei::Mat3x4>,
		unique_device_ptr<Device::CUDA, ei::Mat3x4>> m_instTransformsDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, u32>,
		unique_device_ptr<Device::CUDA, u32>> m_instObjIndicesDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, ei::Box>,
		unique_device_ptr<Device::CUDA, ei::Box>> m_objAabbsDesc;

	ei::Box m_boundingBox;


	template< Device dev >
	void load_materials();
};

}} // namespace mufflon::scene