#include "object.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include "iterators.hpp"

namespace mufflon::scene {

Object::Object(std::size_t vertices, std::size_t edges,
			   std::size_t faces, std::size_t spheres) :
	m_meshData(),
	m_sphereData(),
	m_animationFrame(Object::NO_ANIMATION_FRAME),
	m_lodLevel(Object::DEFAULT_LOD_LEVEL)
{
	this->reserve(vertices, edges, faces, spheres);
}

void Object::reserve(std::size_t vertices, std::size_t edges,
					 std::size_t faces, std::size_t spheres) {
	m_meshData.reserve(vertices, edges, faces);
	m_sphereData.reserve(spheres);
}

Object::VertexHandle Object::add_vertex(const Vertex& vertex, const Normal& normal) {
	mAssert(m_meshData.has_vertex_normals());
	VertexHandle vh = m_meshData.add_vertex(vertex);
	this->set_normal(vh, normal);
	return vh;
}

const Object::Vertex& Object::get_vertex(const VertexHandle& handle) const {
	mAssert(handle.is_valid() && handle.idx() < m_meshData.n_vertices());
	return m_meshData.point(handle);
}

/// Sets the position of the vertex
void Object::set_vertex(const VertexHandle& handle, const Vertex& vertex) {
	mAssert(handle.is_valid());
	m_meshData.set_point(handle, vertex);
}

const Object::Normal& Object::get_normal(const VertexHandle& handle) const {
	mAssert(handle.is_valid() && handle.idx() < m_meshData.n_vertices());
	mAssert(m_meshData.has_vertex_normals());
	return m_meshData.normal(handle);
}

void Object::set_normal(const VertexHandle& handle, const Normal& normal) {
	mAssert(handle.is_valid() && handle.idx() < m_meshData.n_vertices());
	mAssert(m_mesh_data.has_vertex_normals());
	m_meshData.set_normal(handle, normal);
}

Object::TriangleHandle Object::add_triangle(const Triangle& triangle) {
	mAssert(triangle[0u] < m_meshData.n_vertices()
			&& triangle[1u] < m_meshData.n_vertices()
			&& triangle[2u] < m_meshData.n_vertices());
	return m_meshData.add_face(VertexHandle(triangle[0u]), VertexHandle(triangle[1u]),
						 VertexHandle(triangle[2u]));
}

Object::QuadHandle Object::add_quad(const Quad& quad) {
	mAssert(quad[0u] < m_meshData.n_vertices()
			&& quad[1u] < m_meshData.n_vertices()
			&& quad[2u] < m_meshData.n_vertices()
			&& quad[3u] < m_meshData.n_vertices());
	return m_meshData.add_face(VertexHandle(quad[0u]), VertexHandle(quad[1u]),
						 VertexHandle(quad[2u]), VertexHandle(quad[3u]));
}

Object::SphereHandle Object::add_sphere(const Sphere& sphere) {
	m_sphereData.push_back(sphere);
	mAssert(m_sphereData.size() < std::numeric_limits<Index>::max());
	return static_cast<Index>(m_sphereData.size() - 1u);
}

IteratorRange<Object::PolyIterator<0u>> Object::polygons() const {
	return {
		PolyIterator<0u>::begin(m_meshData),
		PolyIterator<0u>::end(m_meshData)
	};
}

IteratorRange<Object::PolyIterator<3u>> Object::triangles() const {
	return {
		PolyIterator<3u>::begin(m_meshData),
		PolyIterator<3u>::end(m_meshData)
	};
}

IteratorRange<Object::PolyIterator<4u>> Object::quads() const {
	return {
		PolyIterator<4u>::begin(m_meshData),
		PolyIterator<4u>::end(m_meshData)
	};
}

IteratorRange<Object::SphereIterator> Object::spheres() const {
	return {
		m_sphereData.begin(),
		m_sphereData.end()
	};
}

void Object::tessellate_uniform(OpenMesh::Subdivider::Uniform::SubdividerT<PolyMesh, Real>& tessellater, std::size_t divisions) {
	tessellater(m_meshData, divisions);
}

void Object::tessellate_adaptive(OpenMesh::Subdivider::Adaptive::CompositeT<AdaptivePolyMesh>& tessellater, std::size_t divisions) {
	// TODO: how does adaptive tessellation with OpenMesh work?
	throw std::runtime_error("tessellate_adaptive is not implemented yet");
}

Object Object::create_lod(OpenMesh::Decimater::DecimaterT<PolyMesh>& decimater, std::size_t vertices) {
	std::size_t actualCollapses = decimater.decimate_to(vertices);
	//Object lod = e
	
}

void Object::build_bvh() {
	// TODO
	throw std::runtime_error("build_bvh is not implemented yet");
}

} // namespace mufflon::scene