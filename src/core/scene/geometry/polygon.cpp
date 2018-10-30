#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::scene::geometry {

Polygons::VertexHandle Polygons::add(const Point& point, const Normal& normal,
								   const UvCoordinate& uv) {
	mAssert(m_meshData.has_vertex_normals());
	mAssert(m_meshData.has_vertex_texcoords2D());
	VertexHandle vh = m_meshData.add_vertex(util::pun<OpenMesh::Vec3f>(point));
	// TODO: can we do this? Should work, right?
	(*m_pointsAttr.aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec3f>(point);
	(*m_normalsAttr.aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec3f>(normal);
	(*m_uvsAttr.aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec2f>(uv);
	return vh;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle &vh, const Triangle& tri,
									   MaterialIndex idx) {
	mAssert(vh.is_valid());
	mAssert(vh.idx() < m_meshData.n_vertices());
	mAssert(tri[0u] < m_meshData.n_vertices()
			&& tri[1u] < m_meshData.n_vertices()
			&& tri[2u] < m_meshData.n_vertices());
	// TODO: do we need different types or identifications for triangle or quad types?
	FaceHandle hdl = m_meshData.add_face(VertexHandle(tri[0u]), VertexHandle(tri[1u]),
							   VertexHandle(tri[2u]));
	mAssert(hdl.is_valid());

	(*m_matIndexAttr.aquire<>())[hdl.idx()] = idx;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const VertexHandle &vh, const Quad& quad,
								   MaterialIndex idx) {
	mAssert(vh.is_valid());
	mAssert(vh.idx() < m_meshData.n_vertices());
	mAssert(quad[0u] < m_meshData.n_vertices()
			&& quad[1u] < m_meshData.n_vertices()
			&& quad[2u] < m_meshData.n_vertices()
			&& quad[3u] < m_meshData.n_vertices());
	// TODO: do we need different types or identifications for triangle or quad types?
	FaceHandle hdl = m_meshData.add_face(VertexHandle(quad[0u]), VertexHandle(quad[1u]),
							   VertexHandle(quad[2u]), VertexHandle(quad[3u]));
	mAssert(hdl.is_valid());
	(*m_matIndexAttr.aquire<>())[hdl.idx()] = idx;
	return hdl;
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, std::istream& pointStream,
											  std::istream& normalStream, std::istream& uvStream) {
	mAssert(m_meshData.n_vertices() < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	VertexHandle hdl(static_cast<int>(m_meshData.n_vertices()));

	// Resize the attributes prior
	m_attributes.resize(m_attributes.get_size() + count);

	// Read the attributes
	std::size_t readPoints = m_pointsAttr.restore(pointStream, 0u, count);
	std::size_t readNormals = m_normalsAttr.restore(normalStream, 0u, count);
	std::size_t readUvs = m_uvsAttr.restore(uvStream, 0u, count);

	return {hdl, readPoints, readNormals, readUvs};
}

void Polygons::tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<MeshType, Real>& tessellater,
				std::size_t divisions) {
	tessellater(m_meshData, divisions);
	// Flag the entire polygon as dirty
	m_attributes.mark_changed<>();
	logInfo("Uniformly tessellated polygon mesh with ", divisions, " subdivisions");
}
void Polygons::tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
				std::size_t divisions) {
	// TODO
	throw std::runtime_error("Adaptive tessellation isn't implemented yet");
	// Flag the entire polygon as dirty
	m_attributes.mark_changed<>();
	logInfo("Adaptively tessellated polygon mesh with ", divisions, " subdivisions");
}

void Polygons::create_lod(OpenMesh::Decimater::DecimaterT<MeshType>& decimater,
				std::size_t target_vertices) {
	decimater.mesh() = m_meshData;
	std::size_t targetDecimations = m_meshData.n_vertices() - target_vertices;
	std::size_t actualDecimations = decimater.decimate_to(target_vertices);
	// Flag the entire polygon as dirty
	m_attributes.mark_changed<>();
	logInfo("Decimated polygon mesh (", actualDecimations, "/", targetDecimations,
			" decimations performed");
	// TODO: this leaks mesh outside
}

} // namespace mufflon::scene::geometry