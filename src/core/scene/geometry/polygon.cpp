#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>

namespace mufflon::scene::geometry {

// Default construction, creates material-index attribute.
Polygons::Polygons() :
	m_meshData(std::make_unique<PolygonMeshType>()),
	m_vertexAttributes(*m_meshData),
	m_faceAttributes(*m_meshData),
	m_pointsAttrHdl(create_points_handle()),
	m_normalsAttrHdl(create_normals_handle()),
	m_uvsAttrHdl(create_uvs_handle()),
	m_matIndexAttrHdl(create_mat_index_handle()) {
	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min()
	};
}

// Creates polygon from already-created mesh.
Polygons::Polygons(PolygonMeshType&& mesh) :
	m_meshData(std::make_unique<PolygonMeshType>(std::move(mesh))),
	m_vertexAttributes(*m_meshData),
	m_faceAttributes(*m_meshData),
	m_pointsAttrHdl(create_points_handle()),
	m_normalsAttrHdl(create_normals_handle()),
	m_uvsAttrHdl(create_uvs_handle()),
	m_matIndexAttrHdl(create_mat_index_handle()) {
	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min(),
		std::numeric_limits<float>::min()
	};
}

void Polygons::resize(std::size_t vertices, std::size_t edges, std::size_t faces) {
	m_meshData->resize(vertices, edges, faces);
	m_vertexAttributes.resize(vertices);
	m_faceAttributes.resize(faces);
}

Polygons::VertexHandle Polygons::add(const Point& point, const Normal& normal,
								   const UvCoordinate& uv) {
	mAssert(m_meshData->has_vertex_normals());
	mAssert(m_meshData->has_vertex_texcoords2D());
	VertexHandle vh = m_meshData->add_vertex(util::pun<OpenMesh::Vec3f>(point));
	// Resize the attribute and set the vertex data
	m_vertexAttributes.resize(m_vertexAttributes.get_size() + 1u);
	(*get_points().aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec3f>(point);
	(*get_normals().aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec3f>(normal);
	(*get_uvs().aquire<>())[vh.idx()] = util::pun<OpenMesh::Vec2f>(uv);
	// Expand the mesh's bounding box
	m_boundingBox = ei::Box(m_boundingBox, ei::Box(point));
	return vh;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2) {
	mAssert(v0.is_valid() && static_cast<std::size_t>(v0.idx()) < m_meshData->n_vertices());
	mAssert(v1.is_valid() && static_cast<std::size_t>(v1.idx()) < m_meshData->n_vertices());
	mAssert(v2.is_valid() && static_cast<std::size_t>(v2.idx()) < m_meshData->n_vertices());
	mAssert(m_quads == 0u); // To keep the order implicitly
	FaceHandle hdl = m_meshData->add_face(v0, v1, v2);
	mAssert(hdl.is_valid());
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices;
	std::size_t currIndexCount = 3u * m_triangles;
	// TODO: keep track of reserved size to avoid unnecessary reallocs
	Allocator<Device::CPU>::realloc(indexBuffer, currIndexCount, 3u + currIndexCount);
	indexBuffer[currIndexCount + 0u] = static_cast<u32>(v0.idx());
	indexBuffer[currIndexCount + 1u] = static_cast<u32>(v1.idx());
	indexBuffer[currIndexCount + 2u] = static_cast<u32>(v2.idx());
	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_size() + 1u);
	++m_triangles;
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2, MaterialIndex idx) {
	TriangleHandle hdl = this->add(v0, v1, v2);
	(*get_mat_indices().aquire<>())[hdl.idx()] = idx;
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const Triangle& tri) {
	return this->add(VertexHandle(tri[0u]), VertexHandle(tri[1u]), VertexHandle(tri[2u]));
}

Polygons::TriangleHandle Polygons::add(const Triangle& tri, MaterialIndex idx) {
	return this->add(VertexHandle(tri[0u]), VertexHandle(tri[1u]),
					 VertexHandle(tri[2u]), idx);
}

Polygons::TriangleHandle Polygons::add(const std::array<VertexHandle, 3u>& vertices) {
	return this->add(vertices[0u], vertices[1u], vertices[2u]);
}

Polygons::TriangleHandle Polygons::add(const std::array<VertexHandle, 3u>& vertices,
									   MaterialIndex idx) {
	return this->add(vertices[0u], vertices[1u], vertices[2u], idx);
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3) {
	mAssert(v0.is_valid() && static_cast<std::size_t>(v0.idx()) < m_meshData->n_vertices());
	mAssert(v1.is_valid() && static_cast<std::size_t>(v1.idx()) < m_meshData->n_vertices());
	mAssert(v2.is_valid() && static_cast<std::size_t>(v2.idx()) < m_meshData->n_vertices());
	mAssert(v3.is_valid() && static_cast<std::size_t>(v3.idx()) < m_meshData->n_vertices());
	FaceHandle hdl = m_meshData->add_face(v0, v1, v2, v3);
	mAssert(hdl.is_valid());
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices;
	std::size_t currIndexCount = 3u * m_triangles + 4u * m_quads;
	// TODO: keep track of reserved size to avoid unnecessary reallocs
	Allocator<Device::CPU>::realloc(indexBuffer, currIndexCount, 4u + currIndexCount);
	indexBuffer[currIndexCount + 0u] = static_cast<u32>(v0.idx());
	indexBuffer[currIndexCount + 1u] = static_cast<u32>(v1.idx());
	indexBuffer[currIndexCount + 2u] = static_cast<u32>(v2.idx());
	indexBuffer[currIndexCount + 3u] = static_cast<u32>(v3.idx());
	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_size() + 1u);
	++m_quads;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3,
								   MaterialIndex idx) {
	QuadHandle hdl = this->add(v0, v1, v2, v3);
	(*get_mat_indices().aquire<>())[hdl.idx()] = idx;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const Quad& quad) {
	return this->add(VertexHandle(quad[0u]), VertexHandle(quad[1u]),
					 VertexHandle(quad[2u]), VertexHandle(quad[3u]));
}

Polygons::QuadHandle Polygons::add(const Quad& quad, MaterialIndex idx) {
	return this->add(VertexHandle(quad[0u]), VertexHandle(quad[1u]),
					 VertexHandle(quad[2u]), VertexHandle(quad[3u]), idx);
}

Polygons::QuadHandle Polygons::add(const std::array<VertexHandle, 4u>& vertices) {
	return this->add(vertices[0u], vertices[1u], vertices[2u], vertices[3u]);
}

Polygons::QuadHandle Polygons::add(const std::array<VertexHandle, 4u>& vertices,
								   MaterialIndex idx) {
	return this->add(vertices[0u], vertices[1u], vertices[2u],
					 vertices[3u], idx);
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& normalStream, util::IByteReader& uvStream) {
	mAssert(m_meshData->n_vertices() < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	std::size_t start = m_meshData->n_vertices();
	VertexHandle hdl(static_cast<int>(start));

	// Resize the attributes prior
	this->resize(start + count, m_meshData->n_edges(), m_meshData->n_faces());

	// Read the attributes
	std::size_t readPoints = get_points().restore(pointStream, start, count);
	std::size_t readNormals = get_normals().restore(normalStream, start, count);
	std::size_t readUvs = get_uvs().restore(uvStream, start, count);
	// Expand the bounding box
	const OpenMesh::Vec3f* points = *get_points().aquireConst();
	for(std::size_t i = start; i < start + readPoints; ++i) {
		m_boundingBox.max = ei::max(util::pun<ei::Vec3>(points[i]), m_boundingBox.max);
		m_boundingBox.min = ei::min(util::pun<ei::Vec3>(points[i]), m_boundingBox.min);
	}

	return {hdl, readPoints, readNormals, readUvs};
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& normalStream, util::IByteReader& uvStream,
											  const ei::Box& boundingBox) {
	mAssert(m_meshData->n_vertices() < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	std::size_t start = m_meshData->n_vertices();
	VertexHandle hdl(static_cast<int>(start));

	// Resize the attributes prior
	this->resize(start + count, m_meshData->n_edges(), m_meshData->n_faces());

	// Read the attributes
	std::size_t readPoints = get_points().restore(pointStream, start, count);
	std::size_t readNormals = get_normals().restore(normalStream, start, count);
	std::size_t readUvs = get_uvs().restore(uvStream, start, count);
	// Expand the bounding box
	m_boundingBox.max = ei::max(boundingBox.max, m_boundingBox.max);
	m_boundingBox.min = ei::min(boundingBox.min, m_boundingBox.min);

	return { hdl, readPoints, readNormals, readUvs };
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& uvStream) {
	mAssert(m_meshData->n_vertices() < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	std::size_t start = m_meshData->n_vertices();
	VertexHandle hdl(static_cast<int>(start));

	// Resize the attributes prior
	this->resize(start + count, m_meshData->n_edges(), m_meshData->n_faces());

	// Read the attributes
	std::size_t readPoints = get_points().restore(pointStream, start, count);
	std::size_t readUvs = get_uvs().restore(uvStream, start, count);
	// Expand the bounding box
	const OpenMesh::Vec3f* points = *get_points().aquireConst();
	for(std::size_t i = start; i < start + readPoints; ++i) {
		m_boundingBox.max = ei::max(util::pun<ei::Vec3>(points[i]), m_boundingBox.max);
		m_boundingBox.min = ei::min(util::pun<ei::Vec3>(points[i]), m_boundingBox.min);
	}

	return { hdl, readPoints, 0u, readUvs };
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& uvStream, const ei::Box& boundingBox) {
	mAssert(m_meshData->n_vertices() < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	std::size_t start = m_meshData->n_vertices();
	VertexHandle hdl(static_cast<int>(start));

	// Resize the attributes prior
	this->resize(start + count, m_meshData->n_edges(), m_meshData->n_faces());

	// Read the attributes
	std::size_t readPoints = get_points().restore(pointStream, start, count);
	std::size_t readUvs = get_uvs().restore(uvStream, start, count);
	// Expand the bounding box
	m_boundingBox.max = ei::max(boundingBox.max, m_boundingBox.max);
	m_boundingBox.min = ei::min(boundingBox.min, m_boundingBox.min);

	return { hdl, readPoints, 0u, readUvs };
}

void Polygons::tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<PolygonMeshType, Real>& tessellater,
				std::size_t divisions) {
	tessellater(*m_meshData, divisions);
	// TODO: change number of triangles/quads!
	// Flag the entire polygon as dirty
	m_vertexAttributes.mark_changed<>();
	logInfo("Uniformly tessellated polygon mesh with ", divisions, " subdivisions");
}
/*void Polygons::tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
				std::size_t divisions) {
	// TODO
	(void)tessellater;
	throw std::runtime_error("Adaptive tessellation isn't implemented yet");
	// TODO: change number of triangles/quads!
	// Flag the entire polygon as dirty
	m_vertexAttributes.mark_changed<>();
	logInfo("Adaptively tessellated polygon mesh with ", divisions, " subdivisions");
}*/

void Polygons::create_lod(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
				std::size_t target_vertices) {
	decimater.mesh() = *m_meshData;
	std::size_t targetDecimations = m_meshData->n_vertices() - target_vertices;
	std::size_t actualDecimations = decimater.decimate_to(target_vertices);
	// Flag the entire polygon as dirty
	// TODO: change number of triangles/quads!
	m_vertexAttributes.mark_changed<>();
	logInfo("Decimated polygon mesh (", actualDecimations, "/", targetDecimations,
			" decimations performed");
	// TODO: this leaks mesh outside
}

Polygons::VertexAttributeHandle<OpenMesh::Vec3f> Polygons::create_points_handle() {
	OpenMesh::VPropHandleT<OpenMesh::Vec3f> omHandle = m_meshData->points_pph();
	mAssert(omHandle.is_valid());
	OpenMesh::PropertyT<OpenMesh::Vec3f>& pointsProp = m_meshData->property(omHandle);
	VertexAttributeHdl<OpenMesh::Vec3f> customHandle = m_vertexAttributes.add<OpenMesh::Vec3f>(pointsProp.name(), omHandle);
	return { std::move(omHandle), std::move(customHandle) };
}

Polygons::VertexAttributeHandle<OpenMesh::Vec3f> Polygons::create_normals_handle() {
	OpenMesh::VPropHandleT<OpenMesh::Vec3f> omHandle = m_meshData->vertex_normals_pph();
	mAssert(omHandle.is_valid());
	OpenMesh::PropertyT<OpenMesh::Vec3f>& normalsProp = m_meshData->property(omHandle);
	VertexAttributeHdl<OpenMesh::Vec3f> customHandle = m_vertexAttributes.add<OpenMesh::Vec3f>(normalsProp.name(), omHandle);
	return { std::move(omHandle), std::move(customHandle) };
}

Polygons::VertexAttributeHandle<OpenMesh::Vec2f> Polygons::create_uvs_handle() {
	OpenMesh::VPropHandleT<OpenMesh::Vec2f> omHandle = m_meshData->vertex_texcoords2D_pph();
	mAssert(omHandle.is_valid());
	OpenMesh::PropertyT<OpenMesh::Vec2f>& uvsProp = m_meshData->property(omHandle);
	VertexAttributeHdl<OpenMesh::Vec2f> customHandle = m_vertexAttributes.add<OpenMesh::Vec2f>(uvsProp.name(), omHandle);
	return { std::move(omHandle), std::move(customHandle) };
}

Polygons::FaceAttributeHandle<MaterialIndex> Polygons::create_mat_index_handle() {
	return this->request<FaceAttributeHandle<MaterialIndex>>("materialIndex");
}

} // namespace mufflon::scene::geometry