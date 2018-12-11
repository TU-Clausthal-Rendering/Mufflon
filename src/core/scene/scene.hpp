#pragma once

#include "descriptors.hpp"
#include "instance.hpp"
#include "handles.hpp"
#include "types.hpp"
#include "lights/light_tree.hpp"
#include "core/cameras/camera.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/scene/accel_structs/accel_struct.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/object.hpp"
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
		m_lightTree.synchronize<dev>();
		m_media.synchronize<dev, Device::CPU>();
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
	template < Device dev, class... VAttrs, class... FAttrs, class... Attrs >
	SceneDescriptor<dev> get_descriptor(const std::tuple<geometry::Polygons::VAttrDesc<VAttrs>...>& vertexAttribs,
										const std::tuple<geometry::Polygons::FAttrDesc<FAttrs>...>& faceAttribs,
										const std::tuple<geometry::Spheres::AttrDesc<Attrs>...>& sphereAttribs) {
		synchronize<dev>();
		std::vector<ObjectDescriptor<dev>> objectDescs;
		std::vector<InstanceDescriptor<dev>> instanceDescs;
		// We need this to ensure we only create one descriptor per object
		std::unordered_map<Object*, u32> objectDescMap;

		// Create the object and instance descriptors hand-in-hand
		for(InstanceHandle inst : m_instances) {
			instanceDescs.push_back(inst->get_descriptor<dev>());
			// Create the object descriptor, if not already present, and its index
			Object* objHdl = &inst->get_object();
			auto entry = objectDescMap.find(objHdl);
			if(entry == objectDescMap.end()) {
				entry = objectDescMap.emplace(objHdl, static_cast<u32>(objectDescs.size())).first;
				objectDescs.push_back(objHdl->get_descriptor<dev>(vertexAttribs, faceAttribs, sphereAttribs));
			}
			instanceDescs.back().objectIndex = entry->second;
		}
		// Allocate the device memory and copy over the descriptors
		auto& objDevDesc = m_objDevDesc.get<unique_device_ptr<dev, ObjectDescriptor<dev>>>();
		auto& instDevDesc = m_instDevDesc.get<unique_device_ptr<dev, InstanceDescriptor<dev>>>();
		objDevDesc = make_udevptr_array<dev, ObjectDescriptor<dev>>(objectDescs.size());
		instDevDesc = make_udevptr_array<dev, InstanceDescriptor<dev>>(instanceDescs.size());
		Allocator<Device::CPU>::template copy<ObjectDescriptor<dev>, dev>(objDevDesc.get(), objectDescs.data(),
																		  objectDescs.size());
		Allocator<Device::CPU>::template copy<InstanceDescriptor<dev>, dev>(instDevDesc.get(), instanceDescs.data(),
																			instanceDescs.size());

		// Bring the light tree to the device
		// We cannot use the make_unique because the light tree doesn't have a proper copy constructor
		auto& lightDevDesc = m_lightDevDesc.get<unique_device_ptr<dev, lights::LightTree<dev>>>();
		//lightDevDesc = make_udevptr_array<dev, lights::LightTree<dev>>();
		lightDevDesc.reset(reinterpret_cast<lights::LightTree<dev>*>(Allocator<dev>::template alloc_array<char>(sizeof(lights::LightTree<dev>))));
		//Allocator<Device::CPU>::copy<lights::LightTree<dev>, dev>(lightDevDesc.get(), &m_lightTree.aquire_tree<dev>(),
		//														  1u);

		load_materials<dev>();

		return SceneDescriptor<dev>{
			static_cast<u32>(objectDescs.size()),
				static_cast<u32>(instanceDescs.size()),
				m_boundingBox,
				objDevDesc.get(),
				instDevDesc.get(),
				lightDevDesc.get(),
				m_media.acquireConst<dev,materials::Medium>(),
				m_materials.acquireConst<dev,int>()
		};
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
	bool m_accelDirty = false;
	std::unique_ptr<accel_struct::IAccelerationStructure> m_accelStruct = nullptr;

	// Resources for descriptors
	util::TaggedTuple<unique_device_ptr<Device::CPU, ObjectDescriptor<Device::CPU>>,
		unique_device_ptr<Device::CUDA, ObjectDescriptor<Device::CUDA>>> m_objDevDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, InstanceDescriptor<Device::CPU>>,
		unique_device_ptr<Device::CUDA, InstanceDescriptor<Device::CUDA>>> m_instDevDesc;
	util::TaggedTuple<unique_device_ptr<Device::CPU, lights::LightTree<Device::CPU>>,
		unique_device_ptr<Device::CUDA, lights::LightTree<Device::CUDA>>> m_lightDevDesc;

	ei::Box m_boundingBox;


	template< Device dev >
	void load_materials();
};

}} // namespace mufflon::scene