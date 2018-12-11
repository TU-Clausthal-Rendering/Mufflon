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

// Data packet handed out by an object (for e.g. BVH construction)
struct ObjectData {
	std::size_t triangleCount;
	std::size_t quadCount;
	ei::Box aabb;
	util::Range<geometry::Polygons::FaceIterator> faces;
	geometry::Polygons::VertexAttribute<OpenMesh::Vec3f>& faceVertices;
	geometry::Spheres::Attribute<ei::Sphere>& spheres;
};

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

	// Resizes storage of geometry type.
	template < class Geom, class... Args >
	void resize(Args&& ...args) {
		m_geometryData.template get<Geom>().resize(std::forward<Args>(args)...);
	}

	// Adds a primitive (e.g. vertex, triangle, sphere...) to geometry.
	template < class Geom, class... Args >
	auto add(Args&& ...args) {
		auto hdl = m_geometryData.template get<Geom>().add(std::forward<Args>(args)...);
		m_accelDirty = true;
		// Expand the object's bounding box
		m_boundingBox.max = ei::max(m_boundingBox.max, m_geometryData.template get<Geom>().get_bounding_box().max);
		m_boundingBox.min = ei::min(m_boundingBox.min, m_geometryData.template get<Geom>().get_bounding_box().min);
		return hdl;
	}

	template < class Geom, class... Args >
	auto add_bulk(Args&& ...args) {
		auto hdl = m_geometryData.get<Geom>().add_bulk(std::forward<Args>(args)...);
		m_accelDirty = true;
		// Expand the object's bounding box
		m_boundingBox.max = ei::max(m_boundingBox.max, m_geometryData.template get<Geom>().get_bounding_box().max);
		m_boundingBox.min = ei::min(m_boundingBox.min, m_geometryData.template get<Geom>().get_bounding_box().min);
		return hdl;
	}

	// Requests an attribute for the geometry type.
	template < class Geom, class Type >
	auto request(const std::string& name) {
		return m_geometryData.template get<Geom>().template request<Type>(name);
	}
	// Removes an attribute for the geometry type.
	template < class Geom, class AttributeHandle >
	void remove(AttributeHandle& handle) {
		m_geometryData.template get<Geom>().remove(handle);
	}
	// Attempts to find an attribute by name.
	template < class Geom, class T >
	auto find(const std::string& name) {
		return m_geometryData.template get<Geom>().template find<T>(name);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * Since we cannot track what the user does with this attribute, the object must manually be marked as 'dirty'
	 * before the next operation on a different residency.
	 */
	template <  class Geom, class AttributeHandle >
	auto &aquire(const AttributeHandle& attrHandle) {
		return m_geometryData.template get<Geom>().aquire(attrHandle);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * The read-only version of aquire does not need the dirty flag.
	 */
	template < class Geom, class AttributeHandle >
	const auto &aquire(const AttributeHandle& attrHandle) const {
		return m_geometryData.template get<Geom>().aquire(attrHandle);
	}

	// Applies tessellation to the geometry type.
	template < class Geom, class Tessellater, class... Args >
	void tessellate(Tessellater& tessellater, Args&& ...args) {
		m_geometryData.template get<Geom>().tessellate(tessellater, std::forward<Args>(args)...);
		m_accelDirty = true;
	}
	// Creates a new LoD by applying a decimater to the geomtry type.
	template < class Geom, class Decimater, class... Args >
	Object create_lod(Decimater& decimater, Args&& ...args) {
		// TODO: what do we want exactly?
		Object temp(*this);
		temp.m_geometryData.template get<Geom>().create_lod(decimater, std::forward<Args>(args)...);
		temp.m_accelDirty = true;
		return temp;
	}

	// Grants access to the material-index attribute (required for ALL geometry types)
	template < class Geom >
	auto& get_mat_indices() {
		return m_geometryData.template get<Geom>().get_mat_indices();
	}

	// Grants access to the material-index attribute (required for ALL geometry types)
	template < class Geom >
	const auto& get_mat_indices() const {
		return m_geometryData.template get<Geom>().get_mat_indices();
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
	template < Device dev, class... VAttrs, class... FAttrs, class... Attrs >
	ObjectDescriptor<dev> get_descriptor(const std::tuple<geometry::Polygons::VAttrDesc<VAttrs>...>& vertexAttribs,
										 const std::tuple<geometry::Polygons::FAttrDesc<FAttrs>...>& faceAttribs,
										 const std::tuple<geometry::Spheres::AttrDesc<Attrs>...>& sphereAttribs) {
		ObjectDescriptor<dev> desc{
			m_boundingBox,
			m_geometryData.get<geometry::Polygons>().get_descriptor<dev>(vertexAttribs, faceAttribs),
			m_geometryData.get<geometry::Spheres>().get_descriptor<dev>(sphereAttribs),
			ArrayDevHandle_t<dev, void>{}
		};
		// TODO: build object BVH if necessary
		return desc;
	}

	// Checks if the acceleration structure on one of the system parts has been modified.
	bool is_accel_dirty(Device res) const noexcept;

	// Checks whether the object currently has a BVH.
	bool has_accel_structure() const noexcept {
		return m_accelStruct != nullptr;
	}
	// Returns the BVH of this object.
	const accel_struct::IAccelerationStructure& get_accel_structure() const noexcept {
		mAssert(this->has_accel_structure());
		return *m_accelStruct;
	}
	// Clears the BVH of this object.
	void clear_accel_structure();
	// Initializes the acceleration structure to a given implementation.
	template < class Accel, class... Args >
	void set_accel_structure(Args&& ...args) {
		m_accelStruct = std::make_unique<Accel>(std::forward<Args>(args)...);
	}
	// (Re-)builds the acceleration structure.
	void build_accel_structure();

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

	bool m_accelDirty = false;
	std::unique_ptr<accel_struct::IAccelerationStructure> m_accelStruct = nullptr;
	std::size_t m_animationFrame = NO_ANIMATION_FRAME; // Current frame of a possible animation
	std::size_t m_lodLevel = DEFAULT_LOD_LEVEL; // Current level-of-detail
	ObjectFlags m_flags;

	// TODO: how to handle the LoDs?
};

}} // namespace mufflon::scene
