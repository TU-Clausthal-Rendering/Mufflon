#pragma once

#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include "mesh.hpp"
#include "sphere.hpp"
#include "util/vectors.hpp"
#include <cstdint>
#include <vector>

namespace mufflon::scene {

using Triangle = std::array<std::uint32_t, 3u>;
using Quad = std::array<std::uint32_t, 4u>;

using Vertex = util::Vec3;
using Normal = util::Vec3;
using Real = float;



/**
 * Representation of a scene object.
 * It contains the geometric data as well as any custom attribute such as normals, importance, etc.
 * It is also responsible for storing meta-data such as animations and LoD levels.
 */
class Object {
public:
	using VertexHandle = OpenMesh::VertexHandle;
	using TriangleHandle = PolyMesh::FaceHandle;
	using QuadHandle = PolyMesh::FaceHandle;
	using SphereHandle = std::size_t; // TODO: do we need invalid indices?
	template < class Type >
	// TODO: FaceProperties?
	using VertexPropertyHandle = OpenMesh::VPropHandleT<T>;

	Object();

	/**
	 * Adds new vertex to the scene.
	 * Any requested properties are left as default-initialized.
	 */
	VertexHandle addVertex(const Vertex &, const Normal &);
	const Vertex &getVertex(const VertexHandle &) const;
	void setVertex(const VertexHandle &, const Vertex &);

	const Normal &getNormal(const VertexHandle &vertexHandle) const;
	void setNormal(const VertexHandle &vertexHandle, const Normal &);

	/**
	 * Adds new triangle.
	 * Any referenced vertices must have been added prior.
	 */
	TriangleHandle addTriangle(const Triangle &);

	/**
	 * Adds new quad.
	 * Any referenced vertices must have been added prior.
	 */
	QuadHandle addQuad(const Quad &);

	/// Adds new sphere.
	SphereHandle addSphere(const Sphere &);

	const Triangle &getTriangle(const TriangleHandle &) const;
	const Quad &getQuad(const QuadHandle &) const;
	const Sphere &getSphere(const SphereHandle &) const;

	/// Requests a new per-vertex property of type Type.
	template < class Type >
	VertexPropertyHandle<Type> requestProperty(const std::string &name = "<vprops>") {
		VertexPropertyHandle<Type> prop_handle;
		m_mesh_data.add_property(prop_handle, name);
		return prop_handle;
	}

	/// Queries the currently set property value of a vertex.
	template < class Type >
	const Type &getProperty(const VertexHandle &vertexHandle, const VertexPropertyHandle<Type> &propHandle) const {
		return m_mesh_data.property(propHandle, vertexHandle);
	}

	/// Queries the currently set property value of a vertex.
	template < class Type >
	void setProperty(const VertexHandle &vertexHandle, const VertexPropertyHandle<Type> &propHandle, const Type &value) {
		m_mesh_data.property(propHandle, vertexHandle) = value;
	}

	/// Returns the object's animation frame
	std::size_t getAnimationFrame() const noexcept { return m_animation_frame; }
	/// Sets the object's animation frame
	void setAnimationFrame(std::size_t frame) noexcept { m_animation_frame = frame; }

	// TODO: adaptive tessellation
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<PolyMesh, Real> &tessellater);

	// TODO: should normals receive special treatment?
	// TODO: iterator for properties
	// TODO: LoDs, animations

private:
	PolyMesh m_mesh_data; /// Structure representing both triangle and quad data
	std::vector<Sphere> m_sphere_data; /// List of spheres
	std::size_t m_animation_frame;
};

} // namespace mufflon::scene