#pragma once

#include "object.hpp"
#include "materials/material.hpp"

namespace mufflon::scene {

/**
 * Container for all things scene-related.
 * This means it stores objects, cameras, materials etc. However, instances and material mappings
 * are being instantiated by a scene object, created based on information from a scenario.
 */
class WorldContainer {
public:
	Object& create_object() {
		m_objects.emplace_back();
		return m_objects.back();
	}

	Object& add_object(Object&& obj) {
		m_objects.push_back(std::move(obj));
		return m_objects.back();
	}

	/*
	 * Add a ready to use material to the scene. The material must be loaded completely.
	 * Only adding a material does not establish any connection with geometry.
	 * This must be done by calling declare_material_slot (once) and assign_material
	 * (whenever the materaial changes).
	 * material: the complete material, ownership is taken.
	 */
	material::MaterialHandle add_material(std::unique_ptr<material::IMaterial> material);

private:
	// All objects of the world.
	std::vector<Object> m_objects;
	// All materials in the scene.
	std::vector<std::unique_ptr<material::IMaterial>> m_materials;
};

} // namespace mufflon::scene