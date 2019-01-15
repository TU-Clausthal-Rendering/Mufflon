#include "scenario.hpp"
#include "util/log.hpp"
#include "core/scene/materials/material.hpp"
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

void Scenario::remove_point_light(const std::string_view& name) {
	m_pointLightNames.erase(std::remove_if(m_pointLightNames.begin(), m_pointLightNames.end(),
										   [&name, this](const std::string_view& n) {
		if(n == name) {
			m_lightsChanged = true;
			return true;
		}
		return false;
	}), m_pointLightNames.end());
}

void Scenario::remove_spot_light(const std::string_view& name) {
	m_spotLightNames.erase(std::remove_if(m_spotLightNames.begin(), m_spotLightNames.end(),
										   [&name, this](const std::string_view& n) {
		if(n == name) {
			m_lightsChanged = true;
			return true;
		}
		return false;
	}), m_spotLightNames.end());
}

void Scenario::remove_dir_light(const std::string_view& name) {
	m_dirLightNames.erase(std::remove_if(m_dirLightNames.begin(), m_dirLightNames.end(),
										   [&name, this](const std::string_view& n) {
		if(n == name) {
			m_lightsChanged = true;
			return true;
		}
		return false;
	}), m_dirLightNames.end());
}

void Scenario::remove_envmap_light() {
	if(!m_envLightName.empty()) {
		m_envLightName = "";
		m_envmapLightsChanged = true;
	}
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