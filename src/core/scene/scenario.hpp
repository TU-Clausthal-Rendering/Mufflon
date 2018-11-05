#pragma once

#include "object.hpp"
#include "export/dll_export.hpp"
#include "materials/material.hpp"
#include "core/cameras/camera.hpp"
#include <map>
#include <string_view>
#include <tuple>

namespace mufflon::scene {

using ObjectHandle = Object*;
using InstanceHandle = Instance*;
using MaterialHandle = material::IMaterial*;
using CameraHandle = cameras::Camera*;

/**
 * This class represents a scenario, meaning a subset of world features.
 * It contains mappings for instances/objects and materials.
 */
class LIBRARY_API Scenario {
public:
	static constexpr std::size_t NO_CUSTOM_LOD = std::numeric_limits<std::size_t>::max();

	/*
	 * Add a new material entry to the table. The index of the material depends on the
	 * order of declarations and is unchanging for a scene.
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
	}

	// Getter/setter for per-object properties
	bool is_masked(ObjectHandle hdl) const;
	std::size_t get_custom_lod(ObjectHandle hdl) const;
	void mask_object(ObjectHandle hdl);
	void set_custom_lod(ObjectHandle hdl, std::size_t level);


private:
	struct MaterialDesc {
		std::string binaryName;
		MaterialHandle material;
	};

	struct ObjectProperty {
		bool masked = false;
		std::size_t lod = NO_CUSTOM_LOD;
	};

	// Map from binaryName to a material index (may use string_views as keys
	// for lookups -> uses a map).
	std::map<std::string, MaterialIndex, std::less<>> m_materialIndices;
	// Map an index to a material including all its names.
	std::vector<MaterialDesc> m_materialAssignment;
	// TODO: list of light mappings

	std::size_t m_globalLodLevel;
	ei::IVec2 m_resolution;
	CameraHandle m_camera;
	// TODO: material properties

	// Object blacklisting and other custom traits
	std::map<ObjectHandle, ObjectProperty> m_perObjectCustomization;
};

} // namespace mufflon::scene