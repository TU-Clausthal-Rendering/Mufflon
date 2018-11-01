#pragma once

#include "materials/material.hpp"
#include <map>

namespace mufflon::scene {

/**
 * This class represents a scenario, meaning a subset of world features.
 * It contains mappings for instances/objects and materials.
 */
class Scenario {
public:
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

private:
	struct MaterialDesc {
		std::string binaryName;
		material::MaterialHandle material;
	};

	// Map from binaryName to a material index (the key-string_views are views
	// to the stringsstored in m_materialAssignment.
	std::map<std::string, MaterialIndex, std::less<>> m_materialIndices;
	// Map an index to a material including all its names.
	std::vector<MaterialDesc> m_materialAssignment;
};

} // namespace mufflon::scene