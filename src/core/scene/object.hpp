#pragma once

#include "bvh.hpp"
#include "mesh.hpp"
#include "sphere.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "util/log.hpp"
#include <climits>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Forward declarations
namespace OpenMesh {

template < class Type >
struct VPropHandleT;
template < class Type >
struct FPropHandleT;
struct VertexHandle;

namespace Decimater {

template < typename Mesh >
class DecimaterT;

} // namespace Decimater

namespace Subdivider {
namespace Uniform {

template < typename Mesh, typename Real >
class SubdividerT;

} // namespace Uniform

namespace Adaptive {

template < typename Mesh >
class CompositeT;

} // namespace Adaptive
} // namespace Subdivider
} // namespace OpenMesh


namespace mufflon::scene {

// Forward declarations
template < class Iter >
class IteratorRange;
template < class Kernel, std::size_t N >
class PolygonCWIterator;

/**
 * Representation of a scene object.
 * It contains the geometric data as well as any custom attribute such as normals, importance, etc.
 * It is also responsible for storing meta-data such as animations and LoD levels.
 */
class Object {
public:
	// Basic properties
	using Vertex = Vec3f;
	using Normal = Vec3f;
	using UvCoordinate = Vec2f;
	using Index = u32;

	// Supported polygons
	using Triangle = std::array<Index, 3u>;
	using Quad = std::array<Index, 4u>;

	// Handles for accessing data
	using VertexHandle = OpenMesh::VertexHandle;
	using TriangleHandle = PolyMesh::FaceHandle;
	using QuadHandle = PolyMesh::FaceHandle;
	using SphereHandle = Index; // TODO: do we need invalid indices?

	// Property handles
	template < class Type >
	using VertexPropertyHandle = OpenMesh::VPropHandleT<Type>;
	template < class Type >
	using FacePropertyHandle = OpenMesh::FPropHandleT<Type>;
	template < std::size_t N >

	using PolyIterator = PolygonCWIterator<PolyMesh::AttribKernel, N>;
	using SphereIterator = typename std::vector<Sphere>::const_iterator;
	using BulkReturn = std::tuple<VertexHandle, std::size_t, std::size_t, std::size_t>;

	static constexpr std::size_t NO_ANIMATION_FRAME = std::numeric_limits<std::size_t>::max();
	static constexpr std::size_t DEFAULT_LOD_LEVEL = 0u;

	// TODO: layout compatibility?
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Vertex), "Vertex type must be compatible to OpenMesh");
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Normal), "Normal type must be compatible to OpenMesh");

	Object() = default;
	/// Constructs a new object and reserves enough space for the given primitives
	Object(std::size_t vertices, std::size_t edges, std::size_t faces, std::size_t spheres);
	/// Creates a new object from a given poly-mesh
	Object(PolyMesh&& mesh);
	Object(const Object&) = default;
	Object(Object&&) = default;
	Object& operator=(const Object&) = default;
	Object& operator=(Object&&) = default;
	~Object() = default;

	/// Reserves enough space in the data structure to fit at least the specified number of objects.
	void reserve(std::size_t vertices, std::size_t edges, std::size_t faces, std::size_t spheres);

	/// Adds a new vertex with default-initialized custom properties.
	VertexHandle add_vertex(const Vertex&, const Normal&, const UvCoordinate&);
	/**
	 * Bulk-loads n vertices (position and normal).
	 * The method calls 'reserve' to ensure enough space exists.
	 */
	template < class IterV, class IterN, class IterUv >
	BulkReturn add_vertex_bulk(IterV vertexBegin, IterN normalBegin, IterUv uvBegin, std::size_t n) {
		this->reserve(m_meshData.n_vertices() + n, m_meshData.n_edges(), m_meshData.n_faces());
		auto currVertex = vertexBegin;
		auto currNormal = normalBegin;
		IterUv currUv = uvBegin;
		// Save the first vertex handle
		VertexHandle firstVh = m_meshData.vertices_begin() + m_meshData.n_vertices();
		for(std::size_t i = 0u; i < n; ++i)
			this->add_vertex(*(currVertex++), *(currNormal++), *(currUv++));
		return {firstVh, n, n, n};
	}
	/**
	 * Bulk-loads n vertices (position and normal).
	 * The method doesn't reserve space in advance, resulting in more allocations
	 * than the sized equivalent.
	 */
	template < class IterV, class IterN, class IterUv >
	BulkReturn add_vertex_bulk(IterV vertexBegin, IterN normalBegin, IterUv uvBegin, IterV vertexEnd) {
		IterV currVertex = vertexBegin;
		IterN currNormal = normalBegin;
		IterUv currUv = uvBegin;
		// Save the first vertex handle
		VertexHandle firstVh = m_meshData.vertices_begin() + m_meshData.n_vertices();
		// Track number of inserted vertices
		std::size_t addedVertices = 0u;
		for(; vertexBegin != vertexEnd; ++addedVertices)
			this->add_vertex(*(currVertex++), *(currNormal++), *(currUv++));
		return {firstVh, addedVertices, addedVertices, addedVertices};
	}
	/**
	 * Bulk-loads n vertices.
	 * The method calls 'reserve' to ensure enough space exists.
	 * Returns the handle to the first read vertex, the number of read vertices,
	 * normals, and texture coordinates;
	 */
	BulkReturn add_vertex_bulk(std::istream&, std::istream&, std::istream&, std::size_t);
	/// Returns the position of a vertex.
	const Vertex& get_vertex(const VertexHandle& handle) const;
	/// Sets the position of the vertex.
	void set_vertex(const VertexHandle& handle, const Vertex& vertex);

	/// Returns the normal of a vertex.
	const Normal &get_normal(const VertexHandle& handle) const;
	/// Sets the normal of a vertex.
	void set_normal(const VertexHandle& handle, const Normal& normal);

	/// Returns the UV coordniates of a vertex.
	const UvCoordinate& get_texcoord(const VertexHandle& handle) const;
	/// Sets the UV coordnates of a vertex.
	void set_texcoord(const VertexHandle& handle, const UvCoordinate& uv);

	/**
	 * Adds new triangle.
	 * Any referenced vertices must have been added prior.
	 */
	TriangleHandle add_triangle(const Triangle&);
	/**
	 * Adds new quad.
	 * Any referenced vertices must have been added prior.
	 */
	QuadHandle add_quad(const Quad&);
	/// Adds new sphere.
	SphereHandle add_sphere(const Sphere&);

	/// Requests a new per-vertex property of type Type.
	template < class Type >
	VertexPropertyHandle<Type> request_property(const std::string& name) {
		VertexPropertyHandle<Type> prop_handle;
		m_meshData.add_property(prop_handle, name);
		logInfo("Added property '", name, "' to object PLACEHOLDER");
		return prop_handle;
	}

	/// Removes the property from the mesh.
	template < class Type >
	void remove_property(const VertexPropertyHandle<Type>& handle) {
		m_meshData.remove_property(handle);
	}

	/**
	 * Attempts to locate a property by its name.
	 */
	template < class Type >
	std::optional<VertexPropertyHandle<Type>> find_property(const std::string& name) {
		VertexPropertyHandle<Type> propertyHandle;
		if(m_meshData.get_property_handle(propertyHandle, name))
			return propertyHandle;
		return std::nullopt;
	}

	/// Queries the currently set property value of a vertex.
	template < class Type >
	const Type& get_property(const VertexHandle& vertexHandle, const VertexPropertyHandle<Type>& propHandle) const {
		return m_meshData.property(propHandle, vertexHandle);
	}

	/// Queries the currently set property value of a vertex.
	template < class Type >
	void set_property(const VertexHandle& vertexHandle, const VertexPropertyHandle<Type>& propHandle, const Type& value) {
		m_meshData.property(propHandle, vertexHandle) = value;
	}

	/**
	 * Uses uniform subdivision to tessellate the mesh.
	 * Level of subdivision etc. need to be preconfigured.
	 */
	void tessellate_uniform(OpenMesh::Subdivider::Uniform::SubdividerT<PolyMesh, Real>&, std::size_t);

	/**
	 * Uses adaptive subdivision to tessellate the mesh.
	 * Since adaptive tessellation requires additional per-mesh attributes, this method
	 * copies the mesh into a different representation before and after.
	 */
	void tessellate_adaptive(OpenMesh::Subdivider::Adaptive::CompositeT<AdaptivePolyMesh>&, std::size_t);

	/**
	 * Creates a new LoD by applying a decimater to the mesh.
	 * The decimater brings its own error function to decide what to decimate.
	 */
	Object create_lod(OpenMesh::Decimater::DecimaterT<PolyMesh>&, std::size_t);

	/// Returns the object's animation frame
	std::size_t get_animation_frame() const noexcept { return m_animationFrame; }
	/// Sets the object's animation frame
	void set_animation_frame(std::size_t frame) noexcept { m_animationFrame = frame; }

	const Bvh& get_bvh() const noexcept { return m_bvh; }

	void build_bvh();

	// TODO: should normals receive special treatment?

private:
	PolyMesh m_meshData; /// Structure representing both triangle and quad data
	std::vector<Sphere> m_sphereData; /// List of spheres
	// TODO: sphere properties
	std::size_t m_animationFrame; /// Current frame of a possible animation
	std::size_t m_lodLevel; /// Current level-of-detail
	// TODO: how to handle the LoDs?
	Bvh m_bvh;

	// TODO: non-CPU memory
};

} // namespace mufflon::scene