#pragma once

#include "object.hpp"
#include "scenario.hpp"
#include "scene.hpp"
#include "materials/material.hpp"
#include "export/dll_export.hpp"
#include <map>
#include <memory>
#include <vector>

namespace mufflon::scene {

/**
 * Container for all things scene-related.
 * This means it stores objects, cameras, materials etc. However, instances and material mappings
 * are being instantiated by a scene object, created based on information from a scenario.
 */
class LIBRARY_API WorldContainer {
public:
	using ObjectHandle = Object*;
	using ScenarioHandle = Scenario*;
	using SceneHandle = Scene*;
	using InstanceHandle = Instance*;

	// Creates a new, empty object and returns a handle to it
	ObjectHandle create_object();
	// Adds an already created object and takes ownership of it
	ObjectHandle add_object(Object&& obj);
	// Creates a new instance.
	InstanceHandle create_instance(ObjectHandle hdl);
	// Adds a new instance.
	InstanceHandle add_instance(Instance &&instance);
	// Create a new scenario
	ScenarioHandle create_scenario();
	// Add a created scenario and take ownership
	ScenarioHandle add_scenario(Scenario&& scenario);

	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	material::MaterialHandle add_material(std::unique_ptr<material::IMaterial> material);

	/**
	 * Loads the specified scenario.
	 * This destroys the currently loaded scene and overwrites it with a new one.
	 * Returns nullptr if something goes wrong.
	 */
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
	// List of all scenarios available
	std::vector<Scenario> m_scenarios;
	// All materials in the scene.
	std::vector<std::unique_ptr<material::IMaterial>> m_materials;
	

	// TODO: cameras, lights, materials

	// Current scene
	std::unique_ptr<Scene> m_scene = nullptr;
};

} // namespace mufflon::scene