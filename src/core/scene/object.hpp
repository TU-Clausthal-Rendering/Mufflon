#pragma once

#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include "bvh.hpp"
#include "mesh.hpp"
#include "sphere.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include <climits>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

template < class Type >
struct OpenMesh::VPropHandleT;
template < class Type >
struct OpenMesh::FPropHandleT;
struct OpenMesh::VertexHandle;
// TODO: class Vec3f;

namespace mufflon::scene {

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
	using Vertex = Vec3f;
	using Normal = Vec3f;
	using Index = u32;
	using Triangle = std::array<Index, 3u>;
	using Quad = std::array<Index, 4u>;
	using VertexHandle = OpenMesh::VertexHandle;
	using TriangleHandle = PolyMesh::FaceHandle;
	using QuadHandle = PolyMesh::FaceHandle;
	using SphereHandle = Index; // TODO: do we need invalid indices?
	template < class Type >
	using VertexPropertyHandle = OpenMesh::VPropHandleT<Type>;
	template < class Type >
	using FacePropertyHandle = OpenMesh::FPropHandleT<Type>;
	template < std::size_t N >
	using PolyIterator = PolygonCWIterator<PolyMesh::AttribKernel, N>;
	using SphereIterator = typename std::vector<Sphere>::const_iterator;

	static constexpr std::size_t NO_ANIMATION_FRAME = std::numeric_limits<std::size_t>::max();
	static constexpr std::size_t DEFAULT_LOD_LEVEL = 0u;

	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Vertex), "Vertex type must be compatible to OpenMesh");
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Normal), "Normal type must be compatible to OpenMesh");

	Object(std::size_t vertices, std::size_t edges, std::size_t faces, std::size_t spheres);
	Object(const Object&) = default;
	Object(Object&&) = default;
	Object& operator=(const Object&) = default;
	Object& operator=(Object&&) = default;
	~Object() = default;

	/// Reserves enough space in the data structure to fit at least the specified number of objects.
	void reserve(std::size_t vertices, std::size_t edges, std::size_t faces, std::size_t spheres);

	/// Adds a new vertex with default-initialized custom properties.
	VertexHandle add_vertex(const Vertex&, const Normal&);
	/// Returns the position of a vertex.
	const Vertex& get_vertex(const VertexHandle& handle) const;

	/// Sets the position of the vertex.
	void set_vertex(const VertexHandle& handle, const Vertex& vertex);
	/// Returns the normal of a vertex.
	const Normal &get_normal(const VertexHandle& handle) const;
	/// Sets the normal of a vertex.
	void set_normal(const VertexHandle& handle, const Normal& normal);

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

	/// Returns an iterator range for all polygons
	IteratorRange<PolyIterator<0u>> polygons() const;
	/// Returns an iterator range for all triangles
	IteratorRange<PolyIterator<3u>> triangles() const;
	/// Returns an iterator range for all quads
	IteratorRange<PolyIterator<4u>> quads() const;
	/// Returns an iterator range for all spheres
	IteratorRange<SphereIterator> spheres() const;

	/// Requests a new per-vertex property of type Type.
	template < class Type >
	VertexPropertyHandle<Type> request_property(const std::string& name) {
		VertexPropertyHandle<Type> prop_handle;
		m_meshData.add_property(prop_handle, name);
		// TODO: m_meshData.property(proper_handle).reserve()
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
	std::size_t m_animationFrame; /// Current frame of a possible animation
	std::size_t m_lodLevel; /// Current level-of-detail
	// TODO: how to handle the LoDs?
	Bvh m_bvh;
};

} // namespace mufflon::scene