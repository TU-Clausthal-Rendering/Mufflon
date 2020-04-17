#pragma once

#include "object.hpp"
#include "scenario.hpp"
#include "scene.hpp"
#include "handles.hpp"
#include "lights/lights.hpp"
#include "lights/background.hpp"
#include "core/memory/hashmap.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
#include "util/indexed_string_map.hpp"
#include "util/string_pool.hpp"
#include "util/fixed_hashmap.hpp"
#include <map>
#include <memory>
#include <vector>

namespace mufflon {

namespace renderer {
class IRenderer;
} // namespace renderer

namespace scene {

/**
 * Container for all things scene-related.
 * This means it stores objects, cameras, materials etc. However, instances and material mappings
 * are being instantiated by a scene object, created based on information from a scenario.
 */
class WorldContainer {
public:
	using PointLightHandle = std::map<std::string, lights::PointLight, std::less<>>::iterator;
	using SpotLightHandle = std::map<std::string, lights::SpotLight, std::less<>>::iterator;
	using DirLightHandle = std::map<std::string, lights::DirectionalLight, std::less<>>::iterator;
	using EnvLightHandle = std::map<std::string, TextureHandle, std::less<>>::iterator;

	using LodLoadFuncPtr = std::uint32_t(CDECL*)(void* userParams, ObjectHandle obj, u32 lod, u32);
	using ObjMatIndicesFuncPtr = std::uint32_t(CDECL*)(void* userParams, uint32_t objId, uint16_t* matIndices, uint32_t* count);


	enum class Sanity {
		SANE,
		NO_OBJECTS,
		NO_INSTANCES,
		NO_CAMERA,
		NO_LIGHTS
	};

	// --------------------------------------------------------------------------------
	// ------------------------- World container interaction --------------------------
	// --------------------------------------------------------------------------------
	WorldContainer();
	WorldContainer(const WorldContainer&) = delete;
	WorldContainer(WorldContainer&&) = default;
	WorldContainer& operator=(const WorldContainer&) = delete;
	WorldContainer& operator=(WorldContainer&&) = default;
	~WorldContainer() = default;

	// Prepares the world for a fresh load.
	// clear_instance() must be called before (or the world has not been changed yet).
	// Must not be called after the world has been modified.
	void reserve(const u32 objects, const u32 instances);
	// Reserves scenarios. Must have no prior scenarios added
	void reserve(const u32 scenarios);
	// Reserves animation data. Must have no prior bones added.
	void reserve_animation(const u32 numBones, const u32 frameCount);
	// Performs sanity check and marks the end of a loading/modifying process
	Sanity finalize_world(const ei::Box& aabb);
	// Performs a sanity check for a given scenario (respects object masking etc.)
	// while also finalizing it (no more changes allowed)
	Sanity finalize_scenario(ScenarioHandle hdl);
	// Loads the specified scenario.
	// This destroys the currently loaded scene and overwrites it with a new one.
	// Returns nullptr if something goes wrong.
	SceneHandle load_scene(ScenarioHandle hdl, renderer::IRenderer* renderer);
	// Reloads the scene from the current scenario if necessary
	void reload_scene(renderer::IRenderer* renderer);
	// Loads a specific LoD from file, if not already present
	bool load_lod(Object& obj, const u32 lodIndex, const bool asReduced = false);
	// Ejects a specific LoD
	bool unload_lod(Object& obj, const u32 lodIndex);
	// Loads the material indices of an object
	std::vector<MaterialIndex> load_object_material_indices(const u32 objectId);
	std::size_t load_object_material_indices(const u32 objectId, MaterialIndex* buffer);
	// Discards any already applied tessellation/displacement for the current scene
	// and re-tessellates/-displaces with the current max. tessellation level
	void retessellate();


	// --------------------------------------------------------------------------------
	// ------------------------------- Creation methods -------------------------------
	// --------------------------------------------------------------------------------
	ObjectHandle create_object(const StringView name, ObjectFlags flags);
	ObjectHandle duplicate_object(ObjectHandle hdl, const StringView name);
	InstanceHandle create_instance(ObjectHandle hdl, const u32 animationFrame = Instance::NO_ANIMATION_FRAME);
	ScenarioHandle create_scenario(const StringView name);
	/* Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	MaterialHandle add_material(std::unique_ptr<materials::IMaterial> material);
	/* Add a medium to the world. If another medium with the same properties
	 * exists it will be returned and the number of media will not be changed.
	 */
	materials::MediumHandle add_medium(const materials::Medium& medium);
	CameraHandle add_camera(const StringView name, std::unique_ptr<cameras::Camera> camera);
	std::optional<u32> add_light(std::string name, const lights::PointLight& light, const u32 frameCount);
	std::optional<u32> add_light(std::string name, const lights::SpotLight& light, const u32 frameCount);
	std::optional<u32> add_light(std::string name, const lights::DirectionalLight& light, const u32 frameCount);
	std::optional<u32> add_light(std::string name, lights::BackgroundType type);
	TextureHandle add_texture(std::unique_ptr<textures::Texture> texture);
	// Set the transformation of a bone where 'keyframe' is the 0-indexed frame.
	void set_bone(u32 boneIndex, u32 keyframe, const ei::DualQuaternion& transformation);


	// --------------------------------------------------------------------------------
	// ------------------------------- Handle Accessors -------------------------------
	// --------------------------------------------------------------------------------
	// All returned handles remain valid over the lifetime of the world unless otherwise indicated
	ObjectHandle get_object(const StringView name);
	InstanceHandle get_instance(std::size_t index, const u32 animationFrame = Instance::NO_ANIMATION_FRAME);
	const ei::Mat3x4& get_world_to_instance_transformation(ConstInstanceHandle instance) const;
	ei::Mat3x4 compute_instance_to_world_transformation(ConstInstanceHandle instance) const;
	void set_world_to_instance_transformation(ConstInstanceHandle instance, const ei::Mat3x4& mat);
	void set_instance_to_world_transformation(ConstInstanceHandle instance, const ei::Mat3x4& mat);
	ScenarioHandle get_scenario(const StringView name);
	ScenarioHandle get_scenario(std::size_t index);
	ScenarioHandle get_current_scenario() const noexcept;
	SceneHandle get_current_scene();
	bool is_current_scene_valid() const noexcept;
	MaterialHandle get_material(u32 index);
	const materials::Medium& get_medium(materials::MediumHandle hdl) const;
	CameraHandle get_camera(StringView name);
	CameraHandle get_camera(std::size_t index);
	const Bone* get_current_keyframe() const noexcept;
	const Bone* get_keyframe(u32 frame) const;
	lights::PointLight* get_point_light(u32 index, const u32 frame);
	lights::SpotLight* get_spot_light(u32 index, const u32 frame);
	lights::DirectionalLight* get_dir_light(u32 index, const u32 frame);
	lights::Background* get_background(u32 index);
	lights::Background& get_default_background();

	// --------------------------------------------------------------------------------
	// ------------------------------- Count Accessors --------------------------------
	// --------------------------------------------------------------------------------
	std::size_t get_highest_instance_frame() const noexcept;
	std::size_t get_instance_count(const u32 frame) const noexcept;
	std::size_t get_scenario_count() const noexcept;
	std::size_t get_material_count() const noexcept;
	std::size_t get_camera_count() const noexcept { return m_cameras.size(); }
	std::size_t get_point_light_count() const noexcept { return m_pointLights.size(); }
	std::size_t get_spot_light_count() const noexcept { return m_spotLights.size(); }
	std::size_t get_dir_light_count() const noexcept { return m_dirLights.size(); }
	std::size_t get_env_light_count() const noexcept { return m_envLights.size(); }
	std::size_t get_point_light_segment_count(u32 index);
	std::size_t get_spot_light_segment_count(u32 index);
	std::size_t get_dir_light_segment_count(u32 index);
	u32 get_num_bones() const noexcept;
	u32 get_frame_count() const noexcept;
	u32 get_frame_current() const noexcept;
	float get_tessellation_level() const noexcept { return m_tessLevel; }

	// --------------------------------------------------------------------------------
	// ------------------------------- Find and search --------------------------------
	// --------------------------------------------------------------------------------
	StringView get_light_name(u32 index, lights::LightType type) const;
	std::optional<std::pair<u32, lights::LightType>> find_light(const StringView& name);
	bool has_texture(StringView name) const;
	TextureHandle find_texture(StringView name);

	// --------------------------------------------------------------------------------
	// ---------------------------------- Modifiers -----------------------------------
	// --------------------------------------------------------------------------------
	// Bakes an instance into an object with its transformation applied
	void apply_transformation(InstanceHandle hdl);
	void set_light_name(u32 index, lights::LightType type, StringView name);
	// Replaces the texture of an envmap light; also updates its summed area table
	void replace_envlight_texture(u32 index, TextureHandle replacement);
	void ref_texture(TextureHandle hdl);
	void unref_texture(TextureHandle hdl);
	// Set the new animation frame. Caution: this invalidates the currently loaded scene
	// which must thus be set for any active renderer!
	bool set_frame_current(const u32 frameCurrent);
	void set_lod_loader_function(LodLoadFuncPtr func, ObjMatIndicesFuncPtr matFunc, void* userParams);
	void set_tessellation_level(const float tessLevel) { m_tessLevel = tessLevel; }


	// --------------------------------------------------------------------------------
	// ----------------------------- Remove functionality -----------------------------
	// --------------------------------------------------------------------------------
	void remove_camera(CameraHandle hdl);
	void remove_light(u32 index, lights::LightType type);


	// --------------------------------------------------------------------------------
	// ----------------------------- "Dirty-flag" methods -----------------------------
	// --------------------------------------------------------------------------------
	bool mark_light_dirty(u32 index, lights::LightType type);

private:
	SceneHandle load_scene(Scenario& scenario, renderer::IRenderer* renderer);
	bool load_scene_lights();

	// Function pointer for loading a LoD from a scene
	LodLoadFuncPtr m_loadLod = nullptr;
	ObjMatIndicesFuncPtr m_objMatLoad = nullptr;
	void* m_loadLodUserParams = nullptr;

	// A pool for all object/instance names (keeps references valid until world clear)
	util::StringPool m_namePool;

	// All objects of the world.
	util::FixedHashMap<StringView, Object> m_objects;
	// All instances of the world (careful: reserve MUST have been called
	// before adding instances). First come instances valid for all frames,
	// then successively those present for concrete frames.
	std::vector<Instance> m_instances;
	std::vector<ei::Mat3x4> m_worldToInstanceTrans;
	// Stores the start/end instance indices for each frame
	std::vector<std::pair<u32, u32>> m_frameInstanceIndices;
	ei::Box m_aabb;

	// TODO: for improved heap allocation, this should be a single vector/map
	//std::vector<std::vector<std::unique_ptr<Instance>>> m_animatedInstances;
	// List of all scenarios available (mapped to their names)
	util::FixedHashMap<StringView, Scenario> m_scenarios;
	// All materials in the scene.
	std::vector<std::unique_ptr<materials::IMaterial>> m_materials;
	// All media in the world (all with unique properties)
	std::vector<materials::Medium> m_media;
	// All available cameras mapped to their name.
	std::unordered_map<StringView, std::unique_ptr<cameras::Camera>> m_cameras;
	std::vector<decltype(m_cameras)::iterator> m_cameraHandles;
	// All light sources of the scene
	// TODO: should we group these together as well? Would make sense for many-light-scenarios
	util::IndexedStringMap<std::vector<lights::PointLight>> m_pointLights;
	util::IndexedStringMap<std::vector<lights::SpotLight>> m_spotLights;
	util::IndexedStringMap<std::vector<lights::DirectionalLight>> m_dirLights;
	util::IndexedStringMap<lights::Background> m_envLights;
	// Texture cache
	std::unordered_map<StringView, std::unique_ptr<textures::Texture>> m_textures;
	std::unordered_map<TextureHandle, std::size_t> m_texRefCount; // Counts how many remaining references a texture has

	// Current scene
	ScenarioHandle m_scenario = nullptr;
	std::unique_ptr<Scene> m_scene = nullptr;
	bool m_sceneValid = false;

	// This is only stored to error-check out-of-order animated instances
	u32 m_firstKeyFrame = 0u;
	// Current animation frame and range
	u32 m_frameCount = 0u;
	u32 m_frameCurrent = 0u;
	// All keyframes of all bones (order: k * numBones + b)
	std::vector<Bone> m_animationData;
	u32 m_numBones = 0u;	// Number of bones per keyframe

	// Current tessellation level (levels per pixel)
	float m_tessLevel = 0u;
};

}} // namespace mufflon::scene
