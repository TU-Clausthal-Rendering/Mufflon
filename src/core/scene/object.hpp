#pragma once

#include "geometry/polygon.hpp"
#include "geometry/sphere.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include "util/log.hpp"
#include "util/range.hpp"
#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include "core/scene/descriptors.hpp"
#include <climits>
#include <cstdint>
#include <memory>
#include <string>

namespace mufflon {

// Forward declaration
enum class Device : unsigned char;

namespace scene {

// Forward declaration
namespace accel_struct {
	class IAccelerationStructure;
}

struct ObjectFlags : public util::Flags<u32> {
	static constexpr u32 EMISSIVE = 1u;
};

/**
 * Representation of a scene object.
 * It contains the geometric data as well as any custom attribute such as normals, importance, etc.
 * It is also responsible for storing meta-data such as animations and LoD levels.
 */
class Object {
public:
	// Available geometry types - extend if necessary
	using GeometryTuple = util::TaggedTuple<geometry::Polygons, geometry::Spheres>;
	static constexpr std::size_t NO_ANIMATION_FRAME = std::numeric_limits<std::size_t>::max();
	static constexpr std::size_t DEFAULT_LOD_LEVEL = 0u;

	Object();
	Object(const Object&) = delete;
	Object(Object&&) = default;
	Object& operator=(const Object&) = delete;
	Object& operator=(Object&&) = default;
	~Object();

	// Returns the name of the object (references the string in the object map
	// located in the world container)
	const std::string_view& get_name() const noexcept {
		return m_name;
	}

	// Sets the name of the object (care: since it takes a stringview, the
	// underlying string must NOT be moved/changed)
	void set_name(std::string_view name) noexcept {
		m_name = name;
	}

	void set_flags(ObjectFlags flags) noexcept {
		m_flags = flags;
	}

	// Grants direct access to the mesh data (const only).
	// Valid types for Geom are geometry::Polygons, geometry::Spheres
	template < class Geom >
	const auto& get_geometry() const {
		return m_geometryData.template get<Geom>();
	}
	template < class Geom >
	auto& get_geometry() {
		return m_geometryData.template get<Geom>();
	}

	// Returns the object's animation frame.
	std::size_t get_animation_frame() const noexcept {
		return m_animationFrame;
	}
	// Sets the object's animation frame.
	void set_animation_frame(std::size_t frame) noexcept {
		m_animationFrame = frame;
	}

	// Is there any emissive polygon in this object
	bool is_emissive() const noexcept {
		return m_flags.is_set(ObjectFlags::EMISSIVE);
	}

	// Get the descriptor of the object (including all geometry)
	// Synchronizes implicitly
	template < Device dev, std::size_t N, std::size_t M, std::size_t O >
	ObjectDescriptor<dev> get_descriptor(const std::array<const char*, N> &vertexAttribs,
										 const std::array<const char*, M> &faceAttribs,
										 const std::array<const char*, O> &sphereAttribs) {
		ObjectDescriptor<dev> desc{
			m_geometryData.get<geometry::Polygons>().get_descriptor<dev>(vertexAttribs, faceAttribs),
			m_geometryData.get<geometry::Spheres>().get_descriptor<dev>(sphereAttribs),
			m_accelStruct[get_device_index<dev>()]
		};
		// (Re)build acceleration structure if necessary
		if(is_accel_dirty<dev>()) {
			//accel_struct::build_lbvh_obj(desc, m_boundingBox);
			// TODO call after LBVHBuilder.build refactoring
			m_accelStruct[get_device_index<dev>()] = desc.accelStruct;
		}
		return desc;
	}

	// Checks if the acceleration structure on one of the system parts has been modified.
	template < Device dev >
	bool is_accel_dirty() const noexcept {
		return m_accelStruct[get_device_index<dev>()].type == accel_struct::AccelType::NONE;
	}

	// Checks whether the object currently has a BVH.
	/*bool has_accel_structure() const noexcept {
		return m_accelStruct != nullptr;
	}*/
	// Clears the BVH of this object.
	void clear_accel_structure();

	// Makes the data of the geometric object resident in the memory system
	// Eg. position, normal, uv, material index for poly, position, radius, mat index for sphere...
	template < Device dev >
	void synchronize() {
		m_geometryData.for_each([](auto& elem) {
			elem.template synchronize<dev>();
		});
	}

	// Removes this object's data from the given memory system
	template < Device dev >
	void unload() {
		m_geometryData.for_each([](auto& elem) {
			elem.template unload<dev>();
		});
	}

	// Gets the bounding box of the object
	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}
	// Gets the bounding box of the sub-mesh
	template < class Geom >
	const ei::Box& get_bounding_box() const noexcept {
		return m_geometryData.template get<Geom>().get_bounding_box();
	}

private:
	std::string_view m_name;
	GeometryTuple m_geometryData;
	ei::Box m_boundingBox;

	AccelDescriptor m_accelStruct[NUM_DEVICES];
	std::size_t m_animationFrame = NO_ANIMATION_FRAME; // Current frame of a possible animation
	std::size_t m_lodLevel = DEFAULT_LOD_LEVEL; // Current level-of-detail
	ObjectFlags m_flags;

	// TODO: how to handle the LoDs?
};

}} // namespace mufflon::scene
