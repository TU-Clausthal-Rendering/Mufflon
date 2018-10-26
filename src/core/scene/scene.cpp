#include "scene.hpp"

namespace mufflon::scene {

MaterialIndex Scene::declare_material_slot(std::string_view binaryName) {
	// Catch if this slot was added before
	auto it = m_materialIndices.find(binaryName);
	if(it != m_materialIndices.end()) {
		logError("[Scene::declare_material_slot] Trying to add an already existend material slot: '", binaryName, "', but names must be unique.");
		return it->second;
	}

	// Add new slot
	MaterialIndex newIndex = static_cast<MaterialIndex>(m_materialAssignment.size());
	m_materialAssignment.push_back(MaterialDesc{
		std::string{binaryName}, nullptr
	});
	m_materialIndices.emplace(m_materialAssignment.back().binaryName, newIndex);
	return newIndex;
}

MaterialIndex Scene::get_material_slot_index(std::string_view binaryName) {
	auto it = m_materialIndices.find(binaryName);
	if(it == m_materialIndices.end()) {
		logError("[Scene::get_material_slot_index] Cannot find the material slot '", binaryName, "'");
		return 0;
	}

	return it->second;
}

void Scene::assign_material(MaterialIndex index, material::MaterialHandle material) {
	// TODO
}

material::MaterialHandle Scene::get_assigned_material(MaterialIndex index) const {
	// TODO
	return nullptr;
}

material::MaterialHandle Scene::add_material(std::unique_ptr<material::IMaterial> material) {
	// TODO
	return nullptr;
}

} // namespace mufflon::scene