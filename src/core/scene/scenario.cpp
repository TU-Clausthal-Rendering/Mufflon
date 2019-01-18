#include "scenario.hpp"
#include "world_container.hpp"
#include "util/log.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/lights/lights.hpp"
#include <algorithm>

namespace mufflon::scene {

Scenario::Scenario()
{
	this->remove_background();
}

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
	m_materialAssignmentChanged = true;
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

void Scenario::remove_point_light(u32 light) {
	if(m_pointLights.erase(std::remove(m_pointLights.begin(), m_pointLights.end(), light)) != m_pointLights.end())
		m_lightsChanged = true;
}

void Scenario::remove_spot_light(u32 light) {
	if(m_spotLights.erase(std::remove(m_spotLights.begin(), m_spotLights.end(), light)) != m_spotLights.end())
		m_lightsChanged = true;
}

void Scenario::remove_dir_light(u32 light) {
	if(m_dirLights.erase(std::remove(m_dirLights.begin(), m_dirLights.end(), light)) != m_dirLights.end())
		m_lightsChanged = true;
}

void Scenario::remove_background() {
	if(m_background != 0u)
		m_envmapLightsChanged = true;
	m_background = 0u;
}

bool Scenario::materials_dirty_reset() const {
	bool dirty = m_materialAssignmentChanged;
	for(const auto& matAssign : m_materialAssignment) {
		dirty |= matAssign.material->dirty_reset();
	}
	m_materialAssignmentChanged = false;
	return dirty;
}

} // namespace mufflon::scene