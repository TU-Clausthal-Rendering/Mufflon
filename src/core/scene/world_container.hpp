#pragma once

#include "object.hpp"
#include "scenario.hpp"
#include "scene.hpp"
#include "handles.hpp"
#include "lights/lights.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
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
	using TexCacheHandle = std::map<std::string, textures::Texture, std::less<>>::iterator;

	static constexpr float SUGGESTED_MAX_SCENE_SIZE = 1024.f*1024.f;

	WorldContainer(const WorldContainer&) = delete;
	WorldContainer& operator=(const WorldContainer&) = delete;

	// Create a new object to be filled
	ObjectHandle create_object(std::string name, ObjectFlags flags);
	// Finds an object by its name
	ObjectHandle get_object(const std::string_view& name);
	// Creates a new instance.
	InstanceHandle create_instance(ObjectHandle hdl);
	// Add a created scenario and take ownership
	ScenarioHandle create_scenario(std::string name);
	// Finds a scenario by its name
	ScenarioHandle get_scenario(const std::string_view& name);
	// Get the scenario for which load_scene() was called last.
	ConstScenarioHandle get_current_scenario() const noexcept { return m_scenario; }
	// This is for interfacing - get the number of scenarios and the name of each
	std::size_t get_scenario_count() const noexcept { return m_scenarios.size(); }
	// Gets the scenario name - this reference invalidates when new scenarios are added!
	const std::string& get_scenario_name(std::size_t index);

	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	MaterialHandle add_material(std::unique_ptr<materials::IMaterial> material);

	/*
	 * Add a medium to the world. If another medium with the same properties
	 * exists it will be returned and the number of media will not be changed.
	 */
	materials::MediumHandle add_medium(const materials::Medium& medium);

	// Add a fully specfied camera to the pool of all cameras.
	CameraHandle add_camera(std::string name, std::unique_ptr<cameras::Camera> camera);
	void remove_camera(CameraHandle hdl);

	// Find a camera dependent on its name.
	std::size_t get_camera_count() const noexcept { return m_cameras.size(); }
	CameraHandle get_camera(std::string_view name);
	CameraHandle get_camera(std::size_t index);

	std::size_t get_point_light_count() const noexcept { return m_pointLights.size(); }
	std::size_t get_spot_light_count() const noexcept { return m_spotLights.size(); }
	std::size_t get_dir_light_count() const noexcept { return m_dirLights.size(); }
	std::size_t get_env_light_count() const noexcept { return m_envLights.size(); }

	// Adds a new light to the scene
	std::optional<PointLightHandle> add_light(std::string name, lights::PointLight&& light);
	std::optional<SpotLightHandle> add_light(std::string name, lights::SpotLight&& light);
	std::optional<DirLightHandle> add_light(std::string name, lights::DirectionalLight&& light);
	std::optional<EnvLightHandle> add_light(std::string name, TextureHandle env);
	// Finds a light by name
	std::optional<PointLightHandle> get_point_light(const std::string_view& name);
	std::optional<SpotLightHandle> get_spot_light(const std::string_view& name);
	std::optional<DirLightHandle> get_dir_light(const std::string_view& name);
	std::optional<EnvLightHandle> get_env_light(const std::string_view& name);
	PointLightHandle get_point_light(std::size_t index);
	SpotLightHandle get_spot_light(std::size_t index);
	DirLightHandle get_dir_light(std::size_t index);
	EnvLightHandle get_env_light(std::size_t index);
	void remove_light(lights::PointLight* hdl);
	void remove_light(lights::SpotLight* hdl);
	void remove_light(lights::DirectionalLight* hdl);
	void remove_light(TextureHandle* hdl);
	// Checks the type of a light by name
	bool is_point_light(const std::string_view& name) const;
	bool is_spot_light(const std::string_view& name) const;
	bool is_dir_light(const std::string_view& name) const;
	bool is_env_light(const std::string_view& name) const;

	// Add new textures to the scene
	bool has_texture(std::string_view name) const;
	std::optional<TexCacheHandle> find_texture(std::string_view name);
	std::optional<std::string_view> get_texture_name(TextureHandle hdl) const;
	TexCacheHandle add_texture(std::string_view name, u16 width, u16 height, u16 numLayers,
							   textures::Format format, textures::SamplingMode mode,
							   bool sRgb, std::unique_ptr<u8[]> data);

	// Useful only when storing light names
	std::optional<std::string_view> get_light_name_ref(const std::string_view& name) const noexcept;

	// Singleton, creating our global world object
	static WorldContainer& instance() {
		return s_container;
	}

	/**
	 * Loads the specified scenario.
	 * This destroys the currently loaded scene and overwrites it with a new one.
	 * Returns nullptr if something goes wrong.
	 */
	SceneHandle load_scene(ConstScenarioHandle hdl);

	// Returns the currently loaded scene, if present
	SceneHandle get_current_scene() {
		return m_scene.get();
	}

	// Clears the world object from all resources
	static void clear_instance();

private:
	WorldContainer() = default;
	WorldContainer(WorldContainer&&) = default;
	WorldContainer& operator=(WorldContainer&&) = default;
	~WorldContainer() = default;

	SceneHandle load_scene(const Scenario& scenario);

	// Global container object for everything
	static WorldContainer s_container;

	// All objects of the world.
	std::map<std::string, Object, std::less<>> m_objects;
	// All instances of the world
	std::vector<Instance> m_instances;
	// List of all scenarios available (mapped to their names)
	std::map<std::string, Scenario, std::less<>> m_scenarios;
	// All materials in the scene.
	std::vector<std::unique_ptr<materials::IMaterial>> m_materials;
	// All media in the world (all with unique properties)
	std::vector<materials::Medium> m_media;
	// All available cameras mapped to their name.
	std::map<std::string, std::unique_ptr<cameras::Camera>, std::less<>> m_cameras;
	std::vector<decltype(m_cameras)::iterator> m_cameraHandles;
	// All light sources of the scene
	std::map<std::string, lights::PointLight, std::less<>> m_pointLights;
	std::map<std::string, lights::SpotLight, std::less<>> m_spotLights;
	std::map<std::string, lights::DirectionalLight, std::less<>> m_dirLights;
	std::map<std::string, TextureHandle, std::less<>> m_envLights;
	std::vector<PointLightHandle> m_pointLightHandles;
	std::vector<SpotLightHandle> m_spotLightHandles;
	std::vector<DirLightHandle> m_dirLightHandles;
	std::vector<EnvLightHandle> m_envLightHandles;
	// Texture cache
	std::map<std::string, textures::Texture, std::less<>> m_textures;

	// TODO: cameras, lights, materials

	// Current scene
	ConstScenarioHandle m_scenario = nullptr;
	std::unique_ptr<Scene> m_scene = nullptr;
};

} // namespace mufflon::scene