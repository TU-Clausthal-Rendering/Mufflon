#include "scenario.hpp"
#include "util/log.hpp"

namespace mufflon::scene {

MaterialIndex Scenario::declare_material_slot(std::string_view binaryName) {
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

MaterialIndex Scenario::get_material_slot_index(std::string_view binaryName) {
	auto it = m_materialIndices.find(binaryName);
	if(it == m_materialIndices.end()) {
		logError("[Scene::get_material_slot_index] Cannot find the material slot '", binaryName, "'");
		return 0;
	}

	return it->second;
}

void Scenario::assign_material(MaterialIndex index, material::MaterialHandle material) {
	// TODO: check if a renderer is active?
	m_materialAssignment[index].material = material;
}

material::MaterialHandle Scenario::get_assigned_material(MaterialIndex index) const {
	return m_materialAssignment[index].material;
}

bool Scenario::is_masked(ObjectHandle hdl) const {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		return iter->second.masked;
	return false;
}

std::size_t Scenario::get_custom_lod(ObjectHandle hdl) const {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		return iter->second.lod;
	return NO_CUSTOM_LOD;
}

void Scenario::mask_object(ObjectHandle hdl) {
	m_perObjectCustomization[hdl].masked = true;
}

void Scenario::set_custom_lod(ObjectHandle hdl, std::size_t level) {
	m_perObjectCustomization[hdl].lod = level;
}

} // namespace mufflon::scene