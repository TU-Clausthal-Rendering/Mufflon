#pragma once

#include "descriptors.hpp"
#include "scenario.hpp"
#include "accel_structs/lbvh.hpp"
#include "geometry/polygon.hpp"
#include "geometry/sphere.hpp"
#include "util/assert.hpp"
#include "util/tagged_tuple.hpp"

namespace mufflon {

// Forward declaration
enum class Device : unsigned char;

namespace scene {

// Forward declaration
template < Device dev >
struct LodDescriptor;
class Object;

namespace tessellation {
class TessLevelOracle;
} // namespace tessellation

class Lod {
public:
	// Available geometry types - extend if necessary
	using GeometryTuple = util::TaggedTuple<geometry::Polygons, geometry::Spheres>;

	Lod(const Object* parent) :
		m_geometry{},
		m_accelStruct{},
		m_parent{ parent },
		m_flags{ 0 }
	{}
	// Warning: implicit sync!
	Lod(Lod&) = default;
	Lod(Lod&& obj) = delete;
	Lod& operator=(const Lod&) = delete;
	Lod& operator=(Lod&&) = delete;
	~Lod() = default;

	// Grants direct access to the mesh data (const only).
	// Valid types for Geom are geometry::Polygons, geometry::Spheres
	template < class Geom >
	const auto& get_geometry() const {
		return m_geometry.template get<Geom>();
	}
	template < class Geom >
	auto& get_geometry() {
		return m_geometry.template get<Geom>();
	}

	// Updates the LoD's flags according to the given scenarios
	// Takes an unordered_set so the scratch memmory can be shared between LoDs
	void update_flags(const Scenario& scenario, std::unordered_set<MaterialIndex>& uniqueMatCache);

	// Is there any emissive polygon/sphere in this object
	bool is_emissive(const Scenario& scenario) const noexcept {
		return m_flags & (1llu << static_cast<u64>(2u * scenario.get_index()));
	}
	// Is there any displaced polygon in this object
	bool is_displaced(const Scenario& scenario) const noexcept {
		return m_flags & (1llu << static_cast<u64>(2u * scenario.get_index() + 1u));
	}

	// Get the descriptor of the object (including all geometry, but without attributes)
	// Synchronizes implicitly
	template < Device dev >
	LodDescriptor<dev> get_descriptor();
	// Updates the given descriptor's attribute fields
	template < Device dev >
	void update_attribute_descriptor(LodDescriptor<dev>& descriptor,
									 const std::vector<const char*>& vertexAttribs,
									 const std::vector<const char*>& faceAttribs,
									 const std::vector<const char*>& sphereAttribs);

	// Clears the BVH of this object.
	void clear_accel_structure();

	// Makes the data of the geometric object resident in the memory system
	// Eg. position, normal, uv, material index for poly, position, radius, mat index for sphere...
	template < Device dev >
	void synchronize() {
		m_geometry.for_each([](auto& elem) {
			elem.template synchronize<dev>();
		});
	}

	// Removes this object's data from the given memory system
	template < Device dev >
	void unload() {
		m_geometry.for_each([](auto& elem) {
			elem.template unload<dev>();
		});
	}

	// Gets the bounding box of the object
	ei::Box get_bounding_box() const noexcept {
		ei::Box aabb;
		aabb.min = ei::Vec3{ std::numeric_limits<float>::max() };
		aabb.max = ei::Vec3{ -std::numeric_limits<float>::max() };
		if(m_geometry.template get<geometry::Polygons>().get_vertex_count() > 0u)
			aabb = ei::Box{ aabb, m_geometry.template get<geometry::Polygons>().get_bounding_box() };
		if(m_geometry.template get<geometry::Spheres>().get_sphere_count() > 0u)
			aabb = ei::Box{ aabb, m_geometry.template get<geometry::Spheres>().get_bounding_box() };
		return aabb;
	}

	void set_parent(const Object* parent) noexcept {
		m_parent = parent;
	}

	// Checks if displacement mapping was applied to all of the LoD's geometry
	bool was_displacement_mapping_applied() const noexcept {
		bool wasDisplacementApplied = true;
		m_geometry.for_each([&wasDisplacementApplied](auto& elem) {
			wasDisplacementApplied &= elem.was_displacement_mapping_applied();
		});
		return wasDisplacementApplied;
	}

	// Applies displacement mapping (if not already performed) to the LoD's geometry
	void displace(tessellation::TessLevelOracle& tessellater, const Scenario& scenario);
	// Tessellates the LoD. If scenario is not null, the tessellation is adaptive
	void tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
					const bool usePhong);

private:
	// Geometry data
	GeometryTuple m_geometry;
	// Acceleration structure of the geometry
	accel_struct::LBVHBuilder m_accelStruct;
	const Object* m_parent;
	u64 m_flags;	// Stores flags (2 bits per-scenario)
};

}} // mufflon::scene