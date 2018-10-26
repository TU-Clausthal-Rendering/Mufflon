#pragma once

#include "instance.hpp"
#include "object.hpp"
#include "materials/material.hpp"
#include <memory>
#include <vector>

namespace mufflon::scene {

// Idea: On device side, Materials, Lights, and Cameras call their own functions in a switch statement

/**
 * Main class on the data side.
 * Holds all data related to a scene: materials, cameras and their paths, geometry etc.
 */
class Scene {
public:
	struct DeviceScene {
		Object::DeviceObject* objects;
		//Instance::DeviceInstance* instances;
		// TODO: camera and stuff
	};

	Scene() = default;
	Scene(const Scene&) = default;
	Scene(Scene&&) = default;
	Scene& operator=(const Scene&) = default;
	Scene& operator=(Scene&&) = default;
	~Scene() = default;

	// Creates a new, empty object in the scene.
	Object& create_object();
	// Adds a new instance.
	void add_instance(Instance instance);

	/*
	 * Add a new material entry to the table. The index of the material depends on the
	 * order of dedclarations and is unchanging for a scene.
	 */
	MaterialIndex declare_material_slot(std::string_view binaryName);
	// Get the index of a slot from its name.
	MaterialIndex get_material_slot_index(std::string_view binaryName);
	/*
	 * Assigns a ready loaded material to a material entry.
	 * The assignment can be changed if no renderer is in a running state.
	 * index: the index of the material slot (used in the binary data).
	 * material: The ready to use material
	 */
	void assign_material(MaterialIndex index, material::MaterialHandle material);
	// Find out if and which material is assigned to a slot. Returns nullptr if nothing is assigned.
	material::MaterialHandle get_assigned_material(MaterialIndex index) const;
	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	material::MaterialHandle add_material(std::unique_ptr<material::IMaterial> material);

private:
	struct MaterialDesc {
		std::string binaryName;
		material::MaterialHandle material;
	};
	// TODO: cameras, lights, materials
	std::vector<Object> m_objects;
	std::vector<Instance> m_instances;

	// All materials in the scene.
	std::vector<std::unique_ptr<material::IMaterial>> m_materials;
	// Map an index to a material including all its names.
	std::vector<MaterialDesc> m_materialAssignment;
	// Map from binaryName to a material index (the key-string_views are views
	// to the stringsstored in m_materialAssignment.
	std::map<std::string, MaterialIndex, std::less<>> m_materialIndices;
};

} // namespace mufflon::scene