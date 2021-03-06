﻿#pragma once

#include "instance.hpp"
#include "handles.hpp"
#include "types.hpp"
#include "lights/light_tree.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/generic_resource.hpp"
#include "core/scene/accel_structs/accel_struct_info.hpp"
#include "core/scene/object.hpp"
#include "core/scene/accel_structs/lbvh.hpp"
#include "util/fixed_hashmap.hpp"
#include <memory>
#include <tuple>
#include <vector>

namespace mufflon { namespace scene {

template < Device dev >
struct SceneDescriptor;
class Scenario;
class WorldContainer;

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class Scene {
public:
	struct SceneSizes {
		std::size_t geometrySize = 0u;
		std::size_t instanceSize = 0u;
		std::size_t lodAccelSize = 0u;
		std::size_t instanceAccelSize = 0u;
		std::size_t materialSize = 0u;
		std::size_t textureSize = 0u;
		std::size_t lightSize = 0u;
		std::size_t descriptorOverhead = 0u;
	};

	// Holds offset into instance list as well as number of instances
	// for one object
	struct InstanceRef {
		u32 offset;
		u32 count;
	};

	Scene(WorldContainer& world, const Scenario& scenario, const u32 frame,
		  const ei::Box& aabb, util::FixedHashMap<ObjectHandle, InstanceRef>&& objects,
		  std::vector<InstanceHandle>&& instances,
		  const std::vector<ei::Mat3x4>& worldToInstanceTransformation,
		  const Bone* bones);
	Scene(const Scene&) = delete;
	Scene(Scene&&) = delete;
	Scene& operator=(const Scene&) = delete;
	Scene& operator=(Scene&&) = delete;
	~Scene() = default;

	// Reserves the maximum number of objects to avoid reallocations
	/*void reserve_objects(const u32 count);
	void reserve_instances(const u32 count);
	// Add an instance to be rendered
	void add_instance(InstanceHandle hdl);*/

	void load_media(const std::vector<materials::Medium>& media);

	// Synchronizes entire scene to the device
	template < Device dev >
	void synchronize() {
		// TODO
		/*for(auto& obj : m_objects) {
			for(InstanceHandle instance : obj.second) {
				(void)instance;
			}
		}*/
		m_lightTree.synchronize<dev>(m_boundingBox);
		m_media.synchronize<dev>();
	}

	template < Device dev >
	void unload() {
		/*for(InstanceHandle instance : m_instances) {
			(void)instance;
			// TODO
		}*/
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

	void mark_lights_dirty() {
		m_lightTree.unload<Device::CPU>();
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
					std::vector<lights::DirectionalLight>&& dirLights);
	void set_background(lights::Background& envLightTexture);
	// Discards any already applied tessellation/displacement and re-tessellates/-displaces
	// with the current tessellation level
	bool retessellate(const float tessLevel);

	// Overwrite which camera is used of the scene
	void set_camera(ConstCameraHandle camera) noexcept {
		(void)camera;
		mAssert(camera != nullptr);
		m_cameraDescChanged.for_each([](auto& elem) { elem.changed = true; });
		// TODO: this function is obsolete, once the scene querries the 'changed' flag from the scenario itself.
	}
	// Access the active camera
	ConstCameraHandle get_camera() const noexcept;

	const lights::LightTreeBuilder& get_light_tree_builder() const {
		return m_lightTree;
	}

	// Checks if the scene is sane, ie. if it has lights or emitters, a camera, etc.
	bool is_sane() const noexcept;
	
	/**
	 * Creates a single structure which grants access to all scene data
	 * needed on the specified device.
	 * Synchronizes implicitly.
	 * It needs to be passed up to three tuples, each coupling names and types for
	 * vertex, face, and sphere attributes which the renderer wants to have
	 * access to.
	 * The max. tessellation level determines the max. inner and outer tessellation
	 * level of any face. It may be 0 if no (re-)tessellation is desired.
	 * Optionally takes a function that gets called to load missing LoDs.
	 *
	 * Usage example:
	 * scene::SceneDescriptor<Device::CUDA> sceneDesc = m_currentScene->get_descriptor<Device::CUDA>(
	 *		{ AttributeIdentifier{AttributeType::INT3, "RGB_color"} },
	 *		{}, // No face attributes
	 *		{ AttributeIdentifier{AttributeType::FLOAT, "grayscale"} }
	 * );
	 */
	template < Device dev >
	const SceneDescriptor<dev>& get_descriptor(const std::vector<AttributeIdentifier>& vertexAttribs,
											   const std::vector<AttributeIdentifier>& faceAttribs,
											   const std::vector<AttributeIdentifier>& sphereAttribs,
											   const std::optional<std::function<bool(WorldContainer&, Object&, u32)>> lodLoader = std::nullopt);
	template < Device dev >
	SceneSizes get_estimated_descriptor_size() const noexcept;

	// Get access to the existing objects in the scene (subset from the world)
	const util::FixedHashMap<ObjectHandle, InstanceRef>& get_objects() const noexcept {
		return m_objects;
	}

	const std::vector<InstanceHandle>& get_instances() const noexcept {
		return m_instances;
	}

	// TODO: care!
	u32 remove_instance(ObjectHandle object, const u32 objInstIdx);

	/*
	 * Create a vertex attribute called "mean_curvature" for each polygonal mesh
	 * and compute the curvature values.
	 */
	void compute_curvature();
	/*
	 * Remove the curvature attribute
	 */
	void remove_curvature();

private:
	static constexpr float SUGGESTED_MAX_SCENE_SIZE = 1024.f*1024.f;

	template < Device dev >
	struct ChangedFlag {
		bool changed = true;
	};

	template < Device dev >
	struct AttributeNames {
		std::vector<AttributeIdentifier> lastVertexAttribs;
		std::vector<AttributeIdentifier> lastFaceAttribs;
		std::vector<AttributeIdentifier> lastSphereAttribs;
	};

	template < Device dev >
	void update_camera_medium(SceneDescriptor<dev>& scene);

	WorldContainer& m_world;
	const Scenario& m_scenario;			// Reference to the scenario which is presented by this scene
	const u32 m_frame;					// Frame used for this scene (also path index for camera animation)

	// List of instances and thus objects to-be-rendered
	// We need this to ensure we only create one descriptor per object
	util::FixedHashMap<ObjectHandle, InstanceRef> m_objects;
	// List of instances; object list entries hold an index into this
	std::vector<InstanceHandle> m_instances;
	const std::vector<ei::Mat3x4>& m_worldToInstanceTransformation;
	const Bone* m_bones = nullptr;

	GenericResource m_media;			// Device copy of the media. It is not possible to access the world from a CUDA compiled file.
	//ConstCameraHandle m_camera;		// The single, chosen camera for rendering this scene
	GenericResource m_materials;		// Device instanciation of Material parameter packs and an offset table (first table then data).
	GenericResource m_alphaTextures;	// Alpha texture handles, one for each material

	// Light tree containing all light sources enabled for the scene
	lights::LightTreeBuilder m_lightTree;

	// Acceleration structure over all instances
	accel_struct::LBVHBuilder m_accelStruct;

	// Resources for descriptors
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, LodDescriptor<Device::CPU>[]>,
		unique_device_ptr<Device::CUDA, LodDescriptor<Device::CUDA>[]>,
		unique_device_ptr<NotGl<Device::OPENGL>(), LodDescriptor<Device::OPENGL>[]>> m_lodDevDesc;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, ei::Mat3x4[]>,
		unique_device_ptr<Device::CUDA, ei::Mat3x4[]>,
		unique_device_ptr<Device::OPENGL, ei::Mat3x4[]>> m_instTransformsDesc;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, ei::Mat3x4[]>,
		unique_device_ptr<Device::CUDA, ei::Mat3x4[]>,
		unique_device_ptr<Device::OPENGL, ei::Mat3x4[]>> m_instToWorldTransformsDesc;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, u32[]>,
		unique_device_ptr<Device::CUDA, u32[]>> m_instLodIndicesDesc;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, ei::Box[]>,
		unique_device_ptr<Device::CUDA, ei::Box[]>,
		unique_device_ptr<Device::OPENGL, ei::Box[]>> m_lodAabbsDesc;

	// Descriptor storage
	util::TaggedTuple<
		SceneDescriptor<Device::CPU>, 
		SceneDescriptor<Device::CUDA>,
		SceneDescriptor<Device::OPENGL>> m_descStore;
	// Remember what attributes are part of the descriptor
	util::TaggedTuple<
		AttributeNames<Device::CPU>, 
		AttributeNames<Device::CUDA>,
		AttributeNames<Device::OPENGL>> m_lastAttributeNames;

	// Whether the light tree has changed and needs to fetch its descriptor
	util::TaggedTuple<
		ChangedFlag<Device::CPU>, 
		ChangedFlag<Device::CUDA>,
		ChangedFlag<Device::OPENGL>> m_lightTreeDescChanged;
	// Whether the camera has changed and needs to fetch its descriptor
	util::TaggedTuple<
		ChangedFlag<Device::CPU>, 
		ChangedFlag<Device::CUDA>,
		ChangedFlag<Device::OPENGL>> m_cameraDescChanged;
	// Whether the light tree needs to reevaluate its media; doesn't get set if only
	// the envmap changes
	util::TaggedTuple<
		ChangedFlag<Device::CPU>, 
		ChangedFlag<Device::CUDA>,
		ChangedFlag<Device::OPENGL>> m_lightTreeNeedsMediaUpdate;

	ei::Box m_boundingBox;

	template< Device dev >
	void load_materials();
	
	void update_camera_medium_cpu(SceneDescriptor<Device::CPU>& scene);
};

namespace scene_detail {

void update_camera_medium_cuda(SceneDescriptor<Device::CUDA>& scene);

} // namespace scene_detail

}} // namespace mufflon::scene
