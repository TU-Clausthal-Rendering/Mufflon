#pragma once

#include "residency.hpp"
#include "export/dll_export.hpp"
#include "geometry/polygon.hpp"
#include "geometry/sphere.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "util/log.hpp"
#include "util/tagged_tuple.hpp"
#include "accel_struct.hpp"
#include <climits>
#include <cstdint>
#include <memory>
#include <string>

namespace mufflon::scene {

/**
 * Representation of a scene object.
 * It contains the geometric data as well as any custom attribute such as normals, importance, etc.
 * It is also responsible for storing meta-data such as animations and LoD levels.
 */
class LIBRARY_API Object {
public:
	// Available geometry types - extend if necessary
	using GeometryTuple = util::TaggedTuple<geometry::Polygons, geometry::Spheres>;
	static constexpr std::size_t NO_ANIMATION_FRAME = std::numeric_limits<std::size_t>::max();
	static constexpr std::size_t DEFAULT_LOD_LEVEL = 0u;

	Object() = default;
	Object(const Object&) = delete;
	Object(Object&&) = default;
	Object& operator=(const Object&) = delete;
	Object& operator=(Object&&) = default;
	~Object();

	// Reserves memory for geometry type.
	template < class Geom, class... Args >
	void reserve(Args&& ...args) {
		geometryData.get<Geom>().reserve(std::forward<Args>(args)...);
	}

	// Resizes storage of geometry type.
	template < class Geom, class... Args >
	void resize(Args&& ...args) {
		geometryData.get<Geom>().resize(std::forward<Args>(args)...);
	}

	// Adds a primitive (e.g. vertex, triangle, sphere...) to geometry.
	template < class Geom, class... Args >
	auto add(Args&& ...args) {
		return geometryData.get<Geom>().add(std::forward<Args>(args)...);
	}

	template < class Geom, class... Args >
	auto add_bulk(Args&& ...args) {
		return geometryData.get<Geom>().add_bulk(std::forward<Args>(args)...);
	}

	// Requests an attribute for the geometry type.
	template < class Geom, class Type >
	auto request(const std::string& name) {
		return geometryData.get<Geom>().request(name);
	}
	// Removes an attribute for the geometry type.
	template < class Geom, class AttributeHandle >
	void remove(const AttributeHandle& handle) {
		geometryData.get<Geom>().remove(handle);
	}
	// Attempts to find an attribute by name.
	template < class Geom >
	auto find(const std::string& name) {
		geometryData.get<Geom>().find(name);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * Since we cannot track what the user does with this attribute, the object must manually be marked as 'dirty'
	 * before the next operation on a different residency.
	 */
	template <  class Geom, class AttributeHandle >
	auto &aquire(const AttributeHandle& attrHandle) {
		return geometryData.get<Geom>().aquire(attrHandle);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * The read-only version of aquire does not need the dirty flag.
	 */
	template < class Geom, class AttributeHandle >
	const auto &aquire(const AttributeHandle& attrHandle) const {
		return geometryData.get<Geom>().aquire(attrHandle);
	}

	// Applies tessellation to the geometry type.
	template < class Geom, class Tessellater, class... Args >
	void tessellate(Tessellater& tessellater, Args&& ...args) {
		geometryData.get<Geom>().tessellate(tessellater,std::forward<Args>(args)...);
	}
	// Creates a new LoD by applying a decimater to the geomtry type.
	template < class Geom, class Decimater, class... Args >
	Object create_lod(Decimater& decimater, Args&& ...args) {
		// TODO: what do we want exactly?
		Object temp(*this);
		temp.geometryData.get<Geom>().create_lod(decimater, std::forward<Args>(args)...);
		temp.m_accelDirty = true;
		return temp;
	}

	// Grants access to the material-index attribute (required for ALL geometry types)
	template < class Geom >
	auto& get_mat_indices() {
		return geometryData.get<Geom>().get_mat_indices();
	}

	// Grants access to the material-index attribute (required for ALL geometry types)
	template < class Geom >
	const auto& get_mat_indices() const {
		return geometryData.get<Geom>().get_mat_indices();
	}

	// Grants direct access to the mesh data (const only).
	template < class Geom >
	const auto& get_geometry() {
		return geometryData.get<Geom>();
	}

	// Returns the object's animation frame.
	std::size_t get_animation_frame() const noexcept {
		return m_animationFrame;
	}
	// Sets the object's animation frame.
	void set_animation_frame(std::size_t frame) noexcept {
		m_animationFrame = frame;
	}

	// Checks if data on one of the system parts has been modified.
	bool is_data_dirty(Device res) const noexcept;
	// Checks if the acceleration structure on one of the system parts has been modified.
	bool is_accel_dirty(Device res) const noexcept;
	
	// Checks whether the object currently has a BVH.
	bool has_accel_structure() const noexcept {
		return m_accel_struct != nullptr;
	}
	// Returns the BVH of this object.
	const IAccelerationStructure& get_accel_structure() const noexcept {
		mAssert(this->has_accel_structure());
		return *m_accel_struct;
	}
	// Clears the BVH of this object.
	void clear_accel_structutre();
	// Initializes the acceleration structure to a given implementation.
	template < class Accel, class... Args >
	void set_accel_structure(Args&& ...args) {
		m_accel_struct = std::make_unique<Accel>(std::forward<Args>(args)...);
	}
	// (Re-)builds the acceleration structure.
	void build_accel_structure();

	// Makes the DEFAULT data of the geometric object resident in the memory system
	// Eg. position, normal, uv, material index for poly, position, radius, mat index for sphere...
	template < class Geom, Device dev >
	void synchronize_default() {
		geometryData.get<Geom>().synchronize_default<dev>();
	}

	// Removes this object's data from the given memory system
	void unload_resident(Device);

private:
	GeometryTuple geometryData;

	bool m_accelDirty = false;
	std::size_t m_animationFrame = NO_ANIMATION_FRAME; // Current frame of a possible animation
	std::size_t m_lodLevel = DEFAULT_LOD_LEVEL; // Current level-of-detail
	std::unique_ptr<IAccelerationStructure> m_accel_struct = nullptr;

	// TODO: how to handle the LoDs?
	// TODO: non-CPU memory
};

} // namespace mufflon::scene
