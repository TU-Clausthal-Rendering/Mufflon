#pragma once

#include "residency.hpp"
#include "geometry/polygon.hpp"
#include "geometry/sphere.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "util/log.hpp"
#include "util/tagged_tuple.hpp"
#include <climits>
#include <cstdint>
#include <memory>
#include <string>

namespace mufflon::scene {

// Forward declarations
class IAccellerationStructure;

/**
 * Representation of a scene object.
 * It contains the geometric data as well as any custom attribute such as normals, importance, etc.
 * It is also responsible for storing meta-data such as animations and LoD levels.
 */
class Object {
public:
	// Available geometry types - extend if necessary
	using GeometryTuple = util::TaggedTuple<geometry::Polygons, geometry::Spheres>;

	// Basic properties
	using Point = Vec3f;
	using Normal = Vec3f;
	using UvCoordinate = Vec2f;
	using Index = u32;

	// Property handles
	template < class Type >
	using VertexPropertyHandle = OpenMesh::VPropHandleT<Type>;
	template < class Type >
	using FacePropertyHandle = OpenMesh::FPropHandleT<Type>;

	static constexpr std::size_t NO_ANIMATION_FRAME = std::numeric_limits<std::size_t>::max();
	static constexpr std::size_t DEFAULT_LOD_LEVEL = 0u;

	Object() = default;
	Object(const Object&) = default;
	Object(Object&&) = default;
	Object& operator=(const Object&) = default;
	Object& operator=(Object&&) = default;
	~Object() = default;

	/// Reserves memory for geometry type.
	template < class Geom, class... Args >
	void reserve(Args&& ...args) {
		m_cpuData.m_geometryData.get<Geom>().reserve(std::forward<Args>(args)...);
	}

	/// Resizes storage of geometry type.
	template < class Geom, class... Args >
	void resize(Args&& ...args) {
		m_cpuData.m_isDirty = true;
		m_cpuData.m_geometryData.get<Geom>().resize(std::forward<Args>(args)...);
	}

	/// Adds a primitive (e.g. vertex, triangle, sphere...) to geometry.
	template < class Geom, class... Args >
	auto add(Args&& ...args) {
		m_cpuData.m_isDirty = true;
		return m_cpuData.m_geometryData.get<Geom>().add(std::forward<Args>(args)...);
	}

	/// Requests an attribute for the geometry type.
	template < class Geom, class Type >
	auto request(const std::string& name) {
		m_cpuData.m_isDirty = true;
		return m_cpuData.m_geometryData.get<Geom>().request(name);
	}
	/// Removes an attribute for the geometry type.
	template < class Geom, class AttributeHandle >
	void remove(const AttributeHandle& handle) {
		m_cpuData.m_isDirty = true;
		m_cpuData.m_geometryData.get<Geom>().remove(handle);
	}
	/// Attempts to find an attribute by name.
	template < class Geom >
	auto find(const std::string& name) {
		m_cpuData.m_geometryData.get<Geom>().find(name);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * Since we cannot track what the user does with this attribute, the object must manually be marked as 'dirty'
	 * before the next operation on a different residency.
	 */
	template <  class Geom, class AttributeHandle >
	auto &aquire(const AttributeHandle& attrHandle) {
		return m_cpuData.m_geometryData.get<Geom>().aquire(attrHandle);
	}
	/**
	 * Aquires a reference to an attribute, valid until the attribute gets removed.
	 * The read-only version of aquire does not need the dirty flag.
	 */
	template < class Geom, class AttributeHandle >
	const auto &aquire(const AttributeHandle& attrHandle) const {
		return m_cpuData.m_geometryData.get<Geom>().aquire(attrHandle);
	}

	/// Applies tessellation to the geometry type.
	template < class Geom, class Tessellater, class... Args >
	void tessellate(Tessellater& tessellater, Args&& ...args) {
		m_cpuData.m_isDirty = true;
		m_cpuData.m_geometryData.get<Geom>()
			.tessellate(tessellater,std::forward<Args>(args)...);
	}
	/// Creates a new LoD by applying a decimater to the geomtry type.
	template < class Geom, class Decimater, class... Args >
	Object create_lod(Decimater& decimater, Args&& ...args) {
		// TODO: what do we want exactly?
		Object temp(*this);
		temp.m_cpuData.m_geometryData.get<Geom>()
			.create_lod(decimater, std::forward<Args>(args)...);
		temp.m_cpuData.m_isDirty = true;
		temp.m_accellDirty = true;
		return temp;
	}

	/// Grants direct access to the mesh data (const only).
	template < class Geom, class... Args >
	const auto& native(Args&& ...args) {
		return m_cpuData.m_geometryData.get<Geom>().native(std::forward<Args>(args)...);
	}

	/// Returns the object's animation frame.
	std::size_t get_animation_frame() const noexcept {
		return m_animationFrame;
	}
	/// Sets the object's animation frame.
	void set_animation_frame(std::size_t frame) noexcept {
		m_animationFrame = frame;
	}

	/// Checks if data on one of the system parts has been modified.
	bool is_data_dirty(Residency res) const noexcept;
	/// Checks if the accelleration structure on one of the system parts has been modified.
	bool is_accell_dirty(Residency res) const noexcept;
	
	/// Checks whether the object currently has a BVH.
	bool has_accell_structure() const noexcept {
		return m_accell_struct != nullptr;
	}
	/// Returns the BVH of this object.
	const IAccellerationStructure& get_accell_structure() const noexcept {
		mAssert(this->has_accell_structure());
		return *m_accell_struct;
	}
	/// Clears the BVH of this object.
	void clear_accell_structutre() {
		// Mark as dirty only if we change something
		m_accellDirty |= m_accell_struct != nullptr;
		m_accell_struct.reset();
	}
	/// Initializes the accelleration structure to a given implementation.
	template < class Accell, class... Args >
	void set_accell_structure(Args&& ...args) {
		m_accell_struct = std::make_unique<Accell>(std::forward<Args>(args)...);
	}
	/// (Re-)builds the accelleration structure.
	void build_accell_structure();

	/// Makes this object's data resident in the memory system
	void make_resident(Residency);
	/// Removes this object's data from the given memory system
	void unload_resident(Residency);

private:
	struct {
		GeometryTuple m_geometryData;
		bool m_isDirty;
	} m_cpuData;

	struct {
		// TODO
		bool m_isDirty;
	} m_cudaData;

	struct {
		// TODO
		bool m_isDirty;
	} m_openGlData;

	bool m_accellDirty = false;
	std::size_t m_animationFrame = NO_ANIMATION_FRAME; /// Current frame of a possible animation
	std::size_t m_lodLevel = DEFAULT_LOD_LEVEL; /// Current level-of-detail
	std::unique_ptr<IAccellerationStructure > m_accell_struct = nullptr;

	// TODO: how to handle the LoDs?
	// TODO: non-CPU memory
	// TODO: dirty flags
};

} // namespace mufflon::scene