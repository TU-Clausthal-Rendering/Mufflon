#include "scenario.hpp"
#include "util/log.hpp"
#include <algorithm>

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

MaterialIndex Scenario::get_material_slot_index(std::string_view binaryName) const {
	auto it = m_materialIndices.find(binaryName);
	if(it == m_materialIndices.end()) {
		logError("[Scene::get_material_slot_index] Cannot find the material slot '", binaryName, "'");
		return 0;
	}

	return it->second;
}

void Scenario::assign_material(MaterialIndex index, MaterialHandle material) {
	// TODO: check if a renderer is active?
	m_materialAssignment[index].material = material;
}

MaterialHandle Scenario::get_assigned_material(MaterialIndex index) const {
	return m_materialAssignment[index].material;
}

bool Scenario::is_masked(ConstObjectHandle hdl) const {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		return iter->second.masked;
	return false;
}

std::size_t Scenario::get_custom_lod(ConstObjectHandle hdl) const {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		return (iter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : iter->second.lod;
	return m_globalLodLevel;
}

void Scenario::mask_object(ConstObjectHandle hdl) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		iter->second.masked = true;
	else
		m_perObjectCustomization.insert({ hdl, ObjectProperty{true, NO_CUSTOM_LOD} });
}

void Scenario::set_custom_lod(ConstObjectHandle hdl, std::size_t level) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		iter->second.lod = level;
	else
		m_perObjectCustomization.insert({ hdl, ObjectProperty{false, level} });
}

void Scenario::remove_light(std::size_t index) {
	if(index < m_lightNames.size())
		m_lightNames.erase(m_lightNames.begin() + index);
}

void Scenario::remove_light(const std::string_view& name) {
	m_lightNames.erase(std::remove_if(m_lightNames.begin(), m_lightNames.end(),
		[&name](const std::string_view& n) {
			return n == name;
		}), m_lightNames.end());
}

} // namespace mufflon::scene