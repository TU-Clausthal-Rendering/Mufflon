#pragma once

#include "object.hpp"
#include "scenario.hpp"
#include "scene.hpp"
#include "lights/lights.hpp"
#include "handles.hpp"
#include "export/api.hpp"
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
	using ScenarioHandle = std::map<std::string, Scenario, std::less<>>::iterator;
	using PointLightHandle = std::map<std::string, lights::PointLight, std::less<>>::iterator;
	using SpotLightHandle = std::map<std::string, lights::SpotLight, std::less<>>::iterator;
	using DirLightHandle = std::map<std::string, lights::DirectionalLight, std::less<>>::iterator;
	using EnvLightHandle = std::map<std::string, textures::TextureHandle, std::less<>>::iterator;

	// Create a new object to be filled
	ObjectHandle create_object();
	// Adds an already created object and takes ownership of it
	ObjectHandle add_object(Object&& obj);
	// Creates a new instance.
	InstanceHandle create_instance(ObjectHandle hdl);
	// Adds a new instance.
	InstanceHandle add_instance(Instance &&instance);
	// Add a created scenario and take ownership
	ScenarioHandle add_scenario(Scenario&& scenario);
	// Finds a scenario by its name
	std::optional<ScenarioHandle> get_scenario(const std::string_view& name);

	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	MaterialHandle add_material(std::unique_ptr<materials::IMaterial> material);

	// Add a fully specfied camera to the pool of all cameras.
	CameraHandle add_camera(std::unique_ptr<cameras::Camera> camera);

	// Find a camera dependent on its name.
	CameraHandle get_camera(std::string_view name);

	// Adds a new light to the scene
	std::optional<PointLightHandle> add_light(std::string name, lights::PointLight&& light);
	std::optional<SpotLightHandle> add_light(std::string name, lights::SpotLight&& light);
	std::optional<DirLightHandle> add_light(std::string name, lights::DirectionalLight&& light);
	std::optional<EnvLightHandle> add_light(std::string name, textures::TextureHandle env);
	// Finds a light by name
	std::optional<PointLightHandle> get_point_light(const std::string_view& name);
	std::optional<SpotLightHandle> get_spot_light(const std::string_view& name);
	std::optional<DirLightHandle> get_dir_light(const std::string_view& name);
	std::optional<EnvLightHandle> get_env_light(const std::string_view& name);
	// Checks the type of a light by name
	bool is_point_light(const std::string_view& name) const;
	bool is_spot_light(const std::string_view& name) const;
	bool is_dir_light(const std::string_view& name) const;
	bool is_env_light(const std::string_view& name) const;

	// Useful only when storing light names
	std::optional<std::string_view> get_light_name_ref(const std::string_view& name) const noexcept;

	// Singleton, creating our global world object
	static WorldContainer& instance() {
		static WorldContainer inst;
		return inst;
	}

	/**
	 * Loads the specified scenario.
	 * This destroys the currently loaded scene and overwrites it with a new one.
	 * Returns nullptr if something goes wrong.
	 */
	SceneHandle load_scene(const Scenario& scenario);
	SceneHandle load_scene(ScenarioHandle hdl);

	// Returns the currently loaded scene, if present
	SceneHandle get_current_scene() {
		return m_scene.get();
	}

private:
	// All objects of the world.
	std::vector<Object> m_objects;
	// All instances of the world
	std::vector<Instance> m_instances;
	// List of all scenarios available (mapped to their names)
	std::map<std::string, Scenario, std::less<>> m_scenarios;
	// All materials in the scene.
	std::vector<std::unique_ptr<materials::IMaterial>> m_materials;
	// All available cameras mapped to their name.
	std::unordered_map<std::string_view, std::unique_ptr<cameras::Camera>> m_cameras;
	// All light sources of the scene
	std::map<std::string, lights::PointLight, std::less<>> m_pointLights;
	std::map<std::string, lights::SpotLight, std::less<>> m_spotLights;
	std::map<std::string, lights::DirectionalLight, std::less<>> m_dirLights;
	std::map<std::string, textures::TextureHandle, std::less<>> m_envLights;

	// TODO: cameras, lights, materials

	// Current scene
	std::unique_ptr<Scene> m_scene = nullptr;
};

} // namespace mufflon::scene