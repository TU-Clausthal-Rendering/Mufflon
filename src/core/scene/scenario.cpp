#include "scenario.hpp"
#include "world_container.hpp"
#include "util/log.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/materials/material.hpp"
#include <algorithm>

namespace mufflon::scene {

Scenario::Scenario(util::StringPool& namePool) :
	m_namePool{ namePool }
{
	this->remove_background();
}

MaterialIndex Scenario::declare_material_slot(StringView binaryName) {
	// Catch if this slot was added before
	auto it = m_materialIndices.find(binaryName);
	if(it != m_materialIndices.end()) {
		logError("[Scenario::declare_material_slot] Trying to add an already existend material slot: '", binaryName, "', but names must be unique.");
		return it->second;
	}

	// Add new slot
	MaterialIndex newIndex = static_cast<MaterialIndex>(m_materialAssignment.size());
	const auto pooledName = m_namePool.insert(binaryName);
	m_materialAssignment.push_back(MaterialDesc{
		pooledName, nullptr
	});
	m_materialIndices.emplace(m_materialAssignment.back().binaryName, newIndex);
	return newIndex;
}

MaterialIndex Scenario::get_material_slot_index(StringView binaryName) const {
	auto it = m_materialIndices.find(binaryName);
	if(it == m_materialIndices.end()) {
		logError("[Scenario::get_material_slot_index] Cannot find the material slot '", binaryName, "'");
		return 0;
	}

	return it->second;
}

StringView Scenario::get_material_slot_name(MaterialIndex slotIdx) const {
	return m_materialAssignment.at(slotIdx).binaryName;
}

void Scenario::assign_material(MaterialIndex index, MaterialHandle material) {
	// TODO: check if a renderer is active?
	if(m_materialAssignment[index].material != nullptr && m_materialAssignment[index].material->get_displacement_map() != nullptr) {
		// Since the overwritten material might have been the only displaced one we gotta update the flag
		m_hasDisplacement = false;
		for(std::size_t i = 0u; i < get_num_material_slots(); ++i) {
			if(m_materialAssignment[i].material->get_displacement_map() != nullptr) {
				m_hasDisplacement = true;
				break;
			}
		}
	}
	m_materialAssignment[index].material = material;
	m_materialAssignmentChanged = true;
	m_hasDisplacement |= material->get_displacement_map() != nullptr;
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
				return std::min(m_globalLodLevel, static_cast<u32>(hdl->get_object().get_lod_slot_count() - 1u));
			}
		} else {
			return std::min((instIter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : instIter->second.lod,
							static_cast<u32>(hdl->get_object().get_lod_slot_count() - 1u));
		}
	} else {
		if(auto objIter = m_perObjectCustomization.find(&hdl->get_object()); objIter != m_perObjectCustomization.end()) {
			return std::min((objIter->second.lod == NO_CUSTOM_LOD) ? m_globalLodLevel : objIter->second.lod,
							static_cast<u32>(hdl->get_object().get_lod_slot_count() - 1u));
		} else {
			return std::min(m_globalLodLevel,
							static_cast<u32>(hdl->get_object().get_lod_slot_count() - 1u));
		}
	}
	return std::min(m_globalLodLevel, static_cast<u32>(hdl->get_object().get_lod_slot_count() - 1u));
}

void Scenario::mask_object(ConstObjectHandle hdl) {
	if(auto iter = m_perObjectCustomization.find(hdl); iter != m_perObjectCustomization.end())
		iter->second.masked = true;
	else
		m_perObjectCustomization.insert({ hdl, CustomObjectProperty{{}, true, NO_CUSTOM_LOD} });
}

void Scenario::mask_instance(ConstInstanceHandle hdl) {
	if(auto iter = m_perInstanceCustomization.find(hdl); iter != m_perInstanceCustomization.end())
		iter->second.masked = true;
	else
		m_perInstanceCustomization.insert({ hdl, CustomInstanceProperty{true, NO_CUSTOM_LOD} });
}

std::optional<Scenario::TessellationInfo> Scenario::get_tessellation_info(ConstObjectHandle hdl) const noexcept {
	if(auto iter = m_perObjectCustomization.find(hdl); iter != m_perObjectCustomization.end())
		return iter->second.tessInfo;
	else
		return std::nullopt;
}
void Scenario::set_tessellation_level(ConstObjectHandle hdl, const float level) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter == m_perObjectCustomization.end())
		iter = m_perObjectCustomization.emplace(hdl, CustomObjectProperty{}).first;
	if(!iter->second.tessInfo.has_value())
		iter->second.tessInfo = TessellationInfo{};
	iter->second.tessInfo->level = level;
	m_hasObjectTessellation = true;
}
void Scenario::set_adaptive_tessellation(ConstObjectHandle hdl, const bool value) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter == m_perObjectCustomization.end())
		iter = m_perObjectCustomization.emplace(hdl, CustomObjectProperty{}).first;
	if(!iter->second.tessInfo.has_value())
		iter->second.tessInfo = TessellationInfo{};
	iter->second.tessInfo->adaptive = value;
	m_hasObjectTessellation = true;
}
void Scenario::set_phong_tessellation(ConstObjectHandle hdl, const bool value) {
	auto iter = m_perObjectCustomization.find(hdl);
	if(iter == m_perObjectCustomization.end())
		iter = m_perObjectCustomization.emplace(hdl, CustomObjectProperty{}).first;
	if(!iter->second.tessInfo.has_value())
		iter->second.tessInfo = TessellationInfo{};
	iter->second.tessInfo->usePhong = value;
	m_hasObjectTessellation = true;
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
/*	for(const auto& matAssign : m_materialAssignment) {
		dirty |= matAssign.material->dirty_reset();
	}*/
	m_materialAssignmentChanged = false;
	return dirty;
}

} // namespace mufflon::scene