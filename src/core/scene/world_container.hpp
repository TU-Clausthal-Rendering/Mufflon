#pragma once

#include "object.hpp"
#include "scenario.hpp"
#include "scene.hpp"
#include "handles.hpp"
#include "lights/lights.hpp"
#include "lights/background.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
#include "util/indexed_string_map.hpp"
#include "core/scene/materials/medium.hpp"
#include <map>
#include <memory>
#include <vector>

namespace mufflon::scene {

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

	enum class Sanity {
		SANE,
		NO_OBJECTS,
		NO_INSTANCES,
		NO_CAMERA,
		NO_LIGHTS
	};

	static constexpr float SUGGESTED_MAX_SCENE_SIZE = 1024.f*1024.f;

	WorldContainer(const WorldContainer&) = delete;
	WorldContainer& operator=(const WorldContainer&) = delete;

	// Create a new object to be filled
	ObjectHandle create_object(std::string name, ObjectFlags flags);
	// Finds an object by its name
	ObjectHandle get_object(const StringView& name);
	// Duplicates an object returns handle to the new duplicated object
	ObjectHandle duplicate_object(ObjectHandle hdl, std::string newName);
	// Applies transformation matrix to the object from the instance.
	// Changes the object from the handle.
	void apply_transformation(InstanceHandle hdl);
	// Find an instance by name (for some objects with only one
	// instance both names are equal, but do not need to be).
	// The returned handle is valid over the entire lifetime of the instance.
	InstanceHandle get_instance(const StringView& name);
	// Creates a new instance.
	InstanceHandle create_instance(std::string name, ObjectHandle hdl);
	// This is for interfacing - get the number of instances and the name of each
	std::size_t get_instance_count() const noexcept{ return m_instances.size(); };
	// Gets the instance name - this reference invalidates when new instances are added!
	InstanceHandle get_instance(std::size_t index);
	// Add a created scenario and take ownership
	ScenarioHandle create_scenario(std::string name);
	// Finds a scenario by its name
	ScenarioHandle get_scenario(const StringView& name);
	// Get the scenario for which load_scene() was called last.
	ScenarioHandle get_current_scenario() const noexcept { return m_scenario; }
	// This is for interfacing - get the number of scenarios and the name of each
	std::size_t get_scenario_count() const noexcept { return m_scenarios.size(); }
	// Gets the scenario name - this reference invalidates when new scenarios are added!
	ScenarioHandle get_scenario(std::size_t index);

	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	MaterialHandle add_material(std::unique_ptr<materials::IMaterial> material);

	std::size_t get_material_count() const noexcept {
		return m_materials.size();
	}
	MaterialHandle get_material(i32 index) {
		return m_materials.at(index).get();
	}

	/*
	 * Add a medium to the world. If another medium with the same properties
	 * exists it will be returned and the number of media will not be changed.
	 */
	materials::MediumHandle add_medium(const materials::Medium& medium);

	const materials::Medium& get_medium(materials::MediumHandle hdl) const {
		return m_media.at(hdl);
	}

	// Add a fully specfied camera to the pool of all cameras.
	CameraHandle add_camera(std::string name, std::unique_ptr<cameras::Camera> camera);
	void remove_camera(CameraHandle hdl);

	// Find a camera dependent on its name.
	std::size_t get_camera_count() const noexcept { return m_cameras.size(); }
	CameraHandle get_camera(StringView name);
	CameraHandle get_camera(std::size_t index);

	std::size_t get_point_light_count() const noexcept { return m_pointLights.size(); }
	std::size_t get_spot_light_count() const noexcept { return m_spotLights.size(); }
	std::size_t get_dir_light_count() const noexcept { return m_dirLights.size(); }
	std::size_t get_env_light_count() const noexcept { return m_envLights.size(); }

	// Adds a new light to the scene
	std::optional<u32> add_light(std::string name, lights::PointLight&& light);
	std::optional<u32> add_light(std::string name, lights::SpotLight&& light);
	std::optional<u32> add_light(std::string name, lights::DirectionalLight&& light);
	std::optional<u32> add_light(std::string name, TextureHandle env);

	// Replaces the texture of an envmap light; also updates its summed area table
	void replace_envlight_texture(u32 index, TextureHandle replacement);

	// Finds a light by name
	std::optional<std::pair<u32, lights::LightType>> find_light(const StringView& name);
	// Access the lights properties
	lights::PointLight* get_point_light(u32 index);
	lights::SpotLight* get_spot_light(u32 index);
	lights::DirectionalLight* get_dir_light(u32 index);
	lights::Background* get_background(u32 index);
	// Delete a light using its handle
	void remove_light(u32 index, lights::LightType type);
	// Get the name of a light
	StringView get_light_name(u32 index, lights::LightType type) const;
	void set_light_name(u32 index, lights::LightType type, StringView name);
	// Functions for dirtying cameras and lights
	void mark_camera_dirty(ConstCameraHandle cam);
	void mark_light_dirty(u32 index, lights::LightType type);

	// Add new textures to the scene
	bool has_texture(StringView name) const;
	TextureHandle find_texture(StringView name);
	TextureHandle add_texture(StringView name, u16 width, u16 height, u16 numLayers,
							   textures::Format format, textures::SamplingMode mode,
							   bool sRgb, std::unique_ptr<u8[]> data);
	void ref_texture(TextureHandle hdl);
	void unref_texture(TextureHandle hdl);

	
	// Singleton, creating our global world object
	static WorldContainer& instance() {
		return s_container;
	}

	/**
	 * Loads the specified scenario.
	 * This destroys the currently loaded scene and overwrites it with a new one.
	 * Returns nullptr if something goes wrong.
	 */
	SceneHandle load_scene(ScenarioHandle hdl);
	// Reloads the scene from the current scenario if necessary
	bool reload_scene();

	// Returns the currently loaded scene, if present
	SceneHandle get_current_scene() {
		return m_scene.get();
	}

	// Performs a sanity check on the current world - has lights, cameras etc.
	Sanity is_sane_world() const;
	// Performs a sanity check for a given scenario (respects object masking etc.)
	Sanity is_sane_scenario(ConstScenarioHandle hdl) const;

	// Returns a handle to the background which should be used as default
	lights::Background& get_default_background() {
		static lights::Background defaultBackground = lights::Background::black();
		return defaultBackground;
	}

	// Sets the after-load function for LoDs
	void set_lod_loader_function(bool (CDECL*func)(ObjectHandle, u32)) {
		m_load_lod = func;
	}

	// Clears the world object from all resources
	static void clear_instance();

private:
	WorldContainer();
	WorldContainer(WorldContainer&&) = default;
	WorldContainer& operator=(WorldContainer&&) = default;
	~WorldContainer() = default;

	SceneHandle load_scene(Scenario& scenario);
	bool load_scene_lights();

	// Global container object for everything
	static WorldContainer s_container;

	// Function pointer for loading a LoD from a scene
	bool (CDECL *m_load_lod)(ObjectHandle obj, u32 lod) = nullptr;

	// All objects of the world.
	std::map<std::string, Object, std::less<>> m_objects;
	// All instances of the world
	std::unordered_map<StringView, std::unique_ptr<Instance>> m_instances;
	// List of all scenarios available (mapped to their names)
	std::map<std::string, Scenario, std::less<>> m_scenarios;
	// All materials in the scene.
	std::vector<std::unique_ptr<materials::IMaterial>> m_materials;
	// All media in the world (all with unique properties)
	std::vector<materials::Medium> m_media;
	// All available cameras mapped to their name.
	std::map<std::string, std::unique_ptr<cameras::Camera>, std::less<>> m_cameras;
	std::vector<decltype(m_cameras)::iterator> m_cameraHandles;
	std::unordered_map<ConstCameraHandle, u8> m_camerasDirty;
	// All light sources of the scene
	util::IndexedStringMap<lights::PointLight> m_pointLights;
	util::IndexedStringMap<lights::SpotLight> m_spotLights;
	util::IndexedStringMap<lights::DirectionalLight> m_dirLights;
	util::IndexedStringMap<lights::Background> m_envLights;
	// Dirty flags to keep track of changed values
	bool m_lightsDirty = true;
	bool m_envLightDirty = true;
	// Texture cache
	std::unordered_map<StringView, std::unique_ptr<textures::Texture>> m_textures;
	std::map<TextureHandle, std::size_t> m_texRefCount; // Counts how many remaining references a texture has

	// Current scene
	ScenarioHandle m_scenario = nullptr;
	std::unique_ptr<Scene> m_scene = nullptr;

};

} // namespace mufflon::scene