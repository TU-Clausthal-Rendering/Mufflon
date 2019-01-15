#pragma once

#include "handles.hpp"
#include "types.hpp"
#include <map>
#include <string_view>
#include <vector>

namespace mufflon { namespace scene {

/**
 * This class represents a scenario, meaning a subset of world features.
 * It contains mappings for instances/objects and materials.
 */
class Scenario {
public:
	static constexpr std::size_t NO_CUSTOM_LOD = std::numeric_limits<std::size_t>::max();

	Scenario() = default;
	Scenario(const Scenario&) = delete;
	Scenario(Scenario&&) = default;
	Scenario& operator=(const Scenario&) = delete;
	Scenario& operator=(Scenario&&) = delete;
	~Scenario() = default;

	/*
	 * Add a new material entry to the table. The index of the material depends on the
	 * order of declarations and is unchanging for a scenario.
	 */
	MaterialIndex declare_material_slot(std::string_view binaryName);
	MaterialIndex get_num_material_slots() const noexcept { return static_cast<MaterialIndex>(m_materialAssignment.size()); }
	// Get the index of a slot from its name.
	MaterialIndex get_material_slot_index(std::string_view binaryName) const;
	/*
	 * Assigns a ready loaded material to a material entry.
	 * The assignment can be changed if no renderer is in a running state.
	 * index: the index of the material slot (used in the binary data).
	 * material: The ready to use material
	 */
	void assign_material(MaterialIndex index, MaterialHandle material);
	// Find out if and which material is assigned to a slot. Returns nullptr if nothing is assigned.
	MaterialHandle get_assigned_material(MaterialIndex index) const;

	// Getter/setters for global LoD level
	std::size_t get_global_lod_level() const noexcept {
		return m_globalLodLevel;
	}
	void set_global_lod_level(std::size_t level) noexcept {
		m_globalLodLevel = level;
	}

	// Getter/setters for resolution
	const ei::IVec2& get_resolution() const noexcept {
		return m_resolution;
	}
	void set_resolution(ei::IVec2 res) noexcept {
		m_resolution = res;
	}

	// Getter/setters for camera name
	CameraHandle get_camera() const noexcept {
		return m_camera;
	}
	void set_camera(CameraHandle camera) noexcept {
		m_camera = camera;
		m_cameraChanged = true;
	}

	// Getter/setter for per-object properties
	bool is_masked(ConstObjectHandle hdl) const;
	std::size_t get_custom_lod(ConstObjectHandle hdl) const;
	void mask_object(ConstObjectHandle hdl);
	void set_custom_lod(ConstObjectHandle hdl, std::size_t level);

	const std::string_view& get_name() const noexcept {
		return m_name;
	}
	void set_name(std::string_view name) noexcept {
		m_name = name;
	}

	// Note: no method to change name! because it is being used as
	// key in worldcontainer
	void add_point_light(std::string_view name) {
		if(std::find(m_pointLightNames.begin(), m_pointLightNames.end(), name) == m_pointLightNames.end()) {
			m_pointLightNames.push_back(name);
			m_lightsChanged = true;
		}
	}
	void add_spot_light(std::string_view name) {
		if(std::find(m_spotLightNames.begin(), m_spotLightNames.end(), name) == m_spotLightNames.end()) {
			m_spotLightNames.push_back(name);
			m_lightsChanged = true;
		}
	}
	void add_dir_light(std::string_view name) {
		if(std::find(m_dirLightNames.begin(), m_dirLightNames.end(), name) == m_dirLightNames.end()) {
			m_dirLightNames.push_back(name);
			m_lightsChanged = true;
		}
	}
	void set_envmap_light(std::string_view name) {
		if(name != m_envLightName) {
			m_envLightName = name;
			m_envmapLightsChanged = true;
		}
	}

	void remove_point_light(const std::string_view& name);
	void remove_spot_light(const std::string_view& name);
	void remove_dir_light(const std::string_view& name);
	void remove_envmap_light();

	// Queries whether lights have been added/removed and resets the flag
	bool lights_dirty_reset() {
		bool dirty = m_lightsChanged;
		m_lightsChanged = false;
		return dirty;
	}
	// Queries whether an envmap light has been added/removed and resets the flag
	bool envmap_lights_dirty_reset() {
		bool dirty = m_envmapLightsChanged;
		m_envmapLightsChanged = false;
		return dirty;
	}
	// Queries whether the camera has been changed and resets the flag
	bool camera_dirty_reset() {
		bool dirty = m_cameraChanged;
		m_cameraChanged = false;
		return dirty;
	}

	const std::vector<std::string_view>& get_point_light_names() const noexcept {
		return m_pointLightNames;
	}
	const std::vector<std::string_view>& get_spot_light_names() const noexcept {
		return m_spotLightNames;
	}
	const std::vector<std::string_view>& get_dir_light_names() const noexcept {
		return m_dirLightNames;
	}
	std::string_view get_envmap_light_name() const noexcept {
		return m_envLightName;
	}

private:
	struct MaterialDesc {
		std::string binaryName;
		MaterialHandle material;
	};

	struct ObjectProperty {
		bool masked = false;
		std::size_t lod = NO_CUSTOM_LOD;
	};

	// "Dirty" flags for rebuilding the scene
	// Two flags exist for lights: one for envmap lights, and one for the rest
	bool m_lightsChanged = true;
	bool m_envmapLightsChanged = true;
	bool m_cameraChanged = true;

	std::string_view m_name;
	// Map from binaryName to a material index (may use string_views as keys
	// for lookups -> uses a map).
	std::map<std::string, MaterialIndex, std::less<>> m_materialIndices;
	// Map an index to a material including all its names.
	std::vector<MaterialDesc> m_materialAssignment;
	// All lights which are enabled in this scenario
	std::vector<std::string_view> m_pointLightNames;
	std::vector<std::string_view> m_spotLightNames;
	std::vector<std::string_view> m_dirLightNames;
	std::string_view m_envLightName;


	std::size_t m_globalLodLevel = 0u;
	ei::IVec2 m_resolution = {};
	CameraHandle m_camera = nullptr;

	// Object blacklisting and other custom traits
	std::map<ConstObjectHandle, ObjectProperty> m_perObjectCustomization;
};

}} // namespace mufflon::scene