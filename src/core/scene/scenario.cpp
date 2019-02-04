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

MaterialIndex Scenario::declare_material_slot(StringView binaryName) {
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

MaterialIndex Scenario::get_material_slot_index(StringView binaryName) const {
	auto it = m_materialIndices.find(binaryName);
	if(it == m_materialIndices.end()) {
		logError("[Scene::get_material_slot_index] Cannot find the material slot '", binaryName, "'");
		return 0;
	}

	return it->second;
}

const std::string& Scenario::get_material_slot_name(MaterialIndex slotIdx) const {
	return m_materialAssignment.at(slotIdx).binaryName;
}

void Scenario::assign_material(MaterialIndex index, MaterialHandle material) {
	// TODO: check if a renderer is active?
	m_materialAssignment[index].material = material;
	m_materialAssignmentChanged = true;
}

MaterialHandle Scenario::get_assigned_material(MaterialIndex index) const {
	return m_materialAssignment[index].material;
}

bool Scenario::is_masked(ConstObjectHandle hdl) const noexcept {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		return iter->second.masked;
	return false;
}

bool Scenario::is_masked(ConstInstanceHandle hdl) const noexcept {
	auto iter = m_perInstanceCustomization.find(hdl);
	if(iter != m_perInstanceCustomization.end())
		return iter->second.masked;
	return false;
}

u32 Scenario::get_custom_lod(ConstInstanceHandle hdl) const noexcept {
	auto instIter = m_perInstanceCustomization.find(hdl);
	if(instIter != m_perInstanceCustomization.end())
		return instIter->second.lod;
	return m_globalLodLevel;
}

u32 Scenario::get_custom_lod(ConstObjectHandle hdl) const noexcept {
	auto objIter = m_perObjectCustomization.find(hdl);
	if(objIter != m_perObjectCustomization.end())
		return objIter->second.lod;
	return m_globalLodLevel;
}

u32 Scenario::get_effective_lod(ConstInstanceHandle hdl) const noexcept {
	// Find out the effective LoD of the instance: if it doesn't have a custom LoD, check the object and then the global LoD
	if(auto instIter = m_perInstanceCustomization.find(hdl); instIter != m_perInstanceCustomization.end()) {
		if(instIter->second.lod == NO_CUSTOM_LOD) {
			if(auto objIter = m_perObjectCustomization.find(&hdl->get_object()); objIter != m_perObjectCustomization.end()) {
				return (objIter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : objIter->second.lod;
			} else {
				return m_globalLodLevel;
			}
		} else {
			return (instIter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : instIter->second.lod;
		}
	} else {
		if(auto objIter = m_perObjectCustomization.find(&hdl->get_object()); objIter != m_perObjectCustomization.end()) {
			return (objIter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : objIter->second.lod;
		} else {
			return m_globalLodLevel;
		}
	}
	return m_globalLodLevel;
}

void Scenario::mask_object(ConstObjectHandle hdl) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter != m_perObjectCustomization.end())
		iter->second.masked = true;
	else
		m_perObjectCustomization.insert({ hdl, CustomProperty{true, NO_CUSTOM_LOD} });
}

void Scenario::mask_instance(ConstInstanceHandle hdl) {
	// TODO
}

void Scenario::set_custom_lod(ConstObjectHandle hdl, u32 level) {
	m_perObjectCustomization[hdl].lod = level;
}

void Scenario::set_custom_lod(ConstInstanceHandle hdl, u32 level) {
	m_perInstanceCustomization[hdl].lod = level;
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