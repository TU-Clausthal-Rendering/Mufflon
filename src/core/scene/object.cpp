#include "object.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include "iterators.hpp"
#include <utility>

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

Object::Object(PolyMesh&& mesh) :
	m_meshData(mesh),
	m_sphereData(),
	m_animationFrame(Object::NO_ANIMATION_FRAME),
	m_lodLevel(Object::DEFAULT_LOD_LEVEL)
{}

void Object::reserve(std::size_t vertices, std::size_t edges,
					 std::size_t faces, std::size_t spheres) {
	m_meshData.reserve(vertices, edges, faces);
	m_sphereData.reserve(spheres);
}

Object::VertexHandle Object::add_vertex(const Vertex& vertex, const Normal& normal,
										const UvCoordinate& uv) {
	mAssert(m_meshData.has_vertex_normals());
	VertexHandle vh = m_meshData.add_vertex(vertex);
	this->set_normal(vh, normal);
	this->set_texcoord(vh, uv);
	return vh;
}

Object::BulkReturn Object::add_vertex_bulk(std::istream& vertices, std::istream& normals,
							 std::istream& uvs, std::size_t n) {
	// Track what vertex we started with
	mAssert(m_meshData.n_vertices() < std::numeric_limits<decltype(std::declval<VertexHandle>().idx())>::max());
	VertexHandle firstVh(static_cast<int>(m_meshData.n_vertices()));
	mAssert(firstVh.is_valid());

	// First we need to make sure that we have enough actual (default-initialized) vertices
	// which will be delegated to the mesh kernel
	m_meshData.resize(m_meshData.n_vertices() + n, m_meshData.n_edges(), m_meshData.n_faces());
	
	// Since we now do have enough vertices, we directly read their position from the file
	vertices.read(reinterpret_cast<char *>(&m_meshData.property(m_meshData.points_pph()).data_vector()[firstVh.idx()]),
				  sizeof(OpenMesh::Vec3f) * n);
	std::size_t readVertices = vertices.gcount() / sizeof(OpenMesh::Vec3f);
	// Next come the normals
	normals.read(reinterpret_cast<char *>(&m_meshData.property(m_meshData.vertex_normals_pph()).data_vector()[firstVh.idx()]),
				 std::min(readVertices, n) * sizeof(OpenMesh::Vec3f));
	std::size_t readNormals = normals.gcount() / sizeof(OpenMesh::Vec3f);
	// And now the texture coordinates
	uvs.read(reinterpret_cast<char *>(&m_meshData.property(m_meshData.vertex_texcoords2D_pph()).data_vector()[firstVh.idx()]),
				  std::min(readNormals, n) * sizeof(OpenMesh::Vec2f));
	std::size_t readUvs = normals.gcount() / sizeof(OpenMesh::Vec2f);

	if(readVertices != readNormals || readNormals != readUvs)
		logWarning("Unequal number of vertices and normals or UV coordinates during bulk load",
				   " of object PLACEHOLDER");
	
	logInfo("Bulk-read ", readVertices, "/", readNormals, "/", readUvs, " vertices/normals/UVs",
			" into object PLACEHOLDER");
	return {firstVh, readVertices, readNormals, readUvs};
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
	mAssert(m_meshData.has_vertex_normals());
	m_meshData.set_normal(handle, normal);
}

const Object::UvCoordinate& Object::get_texcoord(const VertexHandle& handle) const {
	mAssert(handle.is_valid() && handle.idx() < m_meshData.n_vertices());
	mAssert(m_meshData.has_vertex_texcoords2D());
	return m_meshData.texcoord2D(handle);
}

void Object::set_texcoord(const VertexHandle& handle, const UvCoordinate& uv) {
	mAssert(handle.is_valid() && handle.idx() < m_meshData.n_vertices());
	mAssert(m_meshData.has_vertex_texcoords2D());
	m_meshData.set_texcoord2D(handle, uv);
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

void Object::tessellate_uniform(OpenMesh::Subdivider::Uniform::SubdividerT<PolyMesh, Real>& tessellater, std::size_t divisions) {
	tessellater(m_meshData, divisions);
	logInfo("Performed uniform subdivision of object PLACEHOLDER (", divisions, " divisions)");
}

void Object::tessellate_adaptive(OpenMesh::Subdivider::Adaptive::CompositeT<AdaptivePolyMesh>& tessellater, std::size_t divisions) {
	// TODO: how does adaptive tessellation with OpenMesh work?
	throw std::runtime_error("tessellate_adaptive is not implemented yet");
}

Object Object::create_lod(OpenMesh::Decimater::DecimaterT<PolyMesh>& decimater, std::size_t vertices) {
	std::size_t plannedCollapses = m_meshData.n_vertices() - vertices;
	std::size_t actualCollapses = decimater.decimate_to(vertices);
	// TODO: object name?
	logInfo("Created new LoD of object PLACEHOLDER (", actualCollapses, "/",
			plannedCollapses, " collapses performed");
	return Object(std::move(decimater.mesh()));
}

void Object::build_bvh() {
	// TODO
	throw std::runtime_error("build_bvh is not implemented yet");
}

} // namespace mufflon::scene