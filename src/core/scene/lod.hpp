#pragma once

#include "descriptors.hpp"
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

class Lod {
public:
	// Available geometry types - extend if necessary
	using GeometryTuple = util::TaggedTuple<geometry::Polygons, geometry::Spheres>;

	Lod() = default;
	// Warning: implicit sync!
	Lod(Lod&) = default;
	Lod(Lod&& obj) = default;
	Lod& operator=(const Lod&) = delete;
	Lod& operator=(Lod&&) = default;
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

	// Is there any emissive polygon in this object
	// Requires the scenario for the material mapping.
	bool is_emissive(const class Scenario& scenario) const noexcept;

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

private:
	// Geometry data
	GeometryTuple m_geometry;
	// Acceleration structure of the geometry
	accel_struct::LBVHBuilder m_accelStruct;

	// TODO: how to handle the LoDs?
};

}} // mufflon::scene