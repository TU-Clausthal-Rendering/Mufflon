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
	m_pointsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->points_pph())),
	m_normalsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->vertex_normals_pph())),
	m_uvsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec2f>(m_meshData->vertex_texcoords2D_pph())),
	m_matIndicesHdl(m_faceAttributes.add_attribute<u16>(MAT_INDICES_NAME))
{
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
	m_pointsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->points_pph())),
	m_normalsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->vertex_normals_pph())),
	m_uvsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec2f>(m_meshData->vertex_texcoords2D_pph())),
	m_matIndicesHdl(m_faceAttributes.add_attribute<u16>(MAT_INDICES_NAME))
{
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

Polygons::Polygons(Polygons&& poly) :
	m_meshData(std::move(poly.m_meshData)),
	m_vertexAttributes(std::move(poly.m_vertexAttributes)),
	m_faceAttributes(std::move(poly.m_faceAttributes)),
	m_pointsHdl(std::move(poly.m_pointsHdl)),
	m_normalsHdl(std::move(poly.m_normalsHdl)),
	m_uvsHdl(std::move(poly.m_uvsHdl)),
	m_matIndicesHdl(std::move(poly.m_matIndicesHdl)),
	m_indexFlags(std::move(poly.m_indexFlags)),
	m_boundingBox(std::move(poly.m_boundingBox)),
	m_triangles(poly.m_triangles),
	m_quads(poly.m_quads) {
	// Move the index and attribute buffers
	poly.m_indexBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		m_indexBuffer.get<ChangedBuffer>() = buffer;
		buffer.reserved = 0u;
		buffer.indices = ArrayDevHandle_t<ChangedBuffer::DEVICE, u32>{};
	});
	poly.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		m_attribBuffer.get<ChangedBuffer>() = buffer;
		buffer.vertSize = 0u;
		buffer.faceSize = 0u;
		buffer.vertex = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
		buffer.face = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
	});
}

Polygons::~Polygons() {
	m_indexBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		if(buffer.reserved != 0)
			Allocator<ChangedBuffer::DEVICE>::free(buffer.indices, buffer.reserved);
	});
	m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		if(buffer.vertSize != 0)
			Allocator<ChangedBuffer::DEVICE>::free(buffer.vertex, buffer.vertSize);
		if(buffer.faceSize != 0)
			Allocator<ChangedBuffer::DEVICE>::free(buffer.face, buffer.faceSize);
	});
}

std::size_t Polygons::add_bulk(std::string_view name, const VertexHandle& startVertex,
					 std::size_t count, util::IByteReader& attrStream) {
	mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
	return m_vertexAttributes.restore(name, attrStream, static_cast<std::size_t>(startVertex.idx()), count);
}

std::size_t Polygons::add_bulk(std::string_view name, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData->n_vertices());
	return m_faceAttributes.restore(name, attrStream, static_cast<std::size_t>(startFace.idx()), count);
}

std::size_t Polygons::add_bulk(OpenMeshAttributePool<false>::AttributeHandle hdl, const VertexHandle& startVertex,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
	return m_vertexAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startVertex.idx()), count);
}

std::size_t Polygons::add_bulk(OpenMeshAttributePool<true>::AttributeHandle hdl, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData->n_vertices());
	return m_faceAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startFace.idx()), count);
}

void Polygons::reserve(std::size_t vertices, std::size_t edges,
					   std::size_t tris, std::size_t quads) {
	// TODO: reserve attributes
	m_meshData->reserve(vertices, edges, tris + quads);
	m_vertexAttributes.reserve(vertices);
	m_faceAttributes.reserve(tris + quads);
	this->reserve_index_buffer<Device::CPU>(3u * tris + 4u * quads);
}

Polygons::VertexHandle Polygons::add(const Point& point, const Normal& normal,
								   const UvCoordinate& uv) {
	mAssert(m_meshData->has_vertex_normals());
	mAssert(m_meshData->has_vertex_texcoords2D());
	VertexHandle vh = m_meshData->add_vertex(util::pun<OpenMesh::Vec3f>(point));
	// Resize the attribute and set the vertex data
	m_vertexAttributes.resize(m_vertexAttributes.get_attribute_elem_count() + 1u);

	m_vertexAttributes.acquire<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl)[vh.idx()] = util::pun<OpenMesh::Vec3f>(point);
	m_vertexAttributes.acquire<Device::CPU, OpenMesh::Vec3f>(m_normalsHdl)[vh.idx()] = util::pun<OpenMesh::Vec3f>(normal);
	m_vertexAttributes.acquire<Device::CPU, OpenMesh::Vec2f>(m_uvsHdl)[vh.idx()] = util::pun<OpenMesh::Vec2f>(uv);
	m_vertexAttributes.mark_changed(Device::CPU, m_pointsHdl);
	m_vertexAttributes.mark_changed(Device::CPU, m_normalsHdl);
	m_vertexAttributes.mark_changed(Device::CPU, m_uvsHdl);
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
	// Expand the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * (m_triangles + 1u));
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices;
	std::size_t currIndexCount = 3u * m_triangles;
	indexBuffer[currIndexCount + 0u] = static_cast<u32>(v0.idx());
	indexBuffer[currIndexCount + 1u] = static_cast<u32>(v1.idx());
	indexBuffer[currIndexCount + 2u] = static_cast<u32>(v2.idx());
	m_indexFlags.mark_changed(Device::CPU);

	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_triangles;
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2, MaterialIndex idx) {
	TriangleHandle hdl = this->add(v0, v1, v2);
	m_faceAttributes.acquire<Device::CPU, u16>(m_matIndicesHdl)[hdl.idx()] = idx;
	m_faceAttributes.mark_changed(Device::CPU, m_matIndicesHdl);
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
	// Expand the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * (m_quads + 1u));
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices;
	std::size_t currIndexCount = 3u * m_triangles + 4u * m_quads;
	indexBuffer[currIndexCount + 0u] = static_cast<u32>(v0.idx());
	indexBuffer[currIndexCount + 1u] = static_cast<u32>(v1.idx());
	indexBuffer[currIndexCount + 2u] = static_cast<u32>(v2.idx());
	indexBuffer[currIndexCount + 3u] = static_cast<u32>(v3.idx());
	m_indexFlags.mark_changed(Device::CPU);
	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_quads;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3,
								   MaterialIndex idx) {
	QuadHandle hdl = this->add(v0, v1, v2, v3);
	m_faceAttributes.acquire<Device::CPU, u16>(m_matIndicesHdl)[hdl.idx()] = idx;
	m_faceAttributes.mark_changed(Device::CPU, m_matIndicesHdl);
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
	this->reserve(start + count, m_meshData->n_edges(), m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readNormals = m_vertexAttributes.restore(m_normalsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, pointStream, start, count);
	// Expand the bounding box
	const OpenMesh::Vec3f* points = m_vertexAttributes.acquire_const<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl);
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
	this->reserve(start + count, m_meshData->n_edges(), m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readNormals = m_vertexAttributes.restore(m_normalsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, pointStream, start, count);
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
	this->reserve(start + count, m_meshData->n_edges(), m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, pointStream, start, count);
	// Expand the bounding box
	const OpenMesh::Vec3f* points = m_vertexAttributes.acquire_const<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl);
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
	this->reserve(start + count, m_meshData->n_edges(), m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, pointStream, start, count);
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
	m_vertexAttributes.mark_changed(Device::CPU);
	logInfo("Uniformly tessellated polygon mesh with ", divisions, " subdivisions");
}
/*void Polygons::tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
				std::size_t divisions) {
	// TODO
	(void)tessellater;
	throw std::runtime_error("Adaptive tessellation isn't implemented yet");
	// TODO: change number of triangles/quads!
	// Flag the entire polygon as dirty
	m_vertexAttributes.mark_changed(Device::CPU);
	logInfo("Adaptively tessellated polygon mesh with ", divisions, " subdivisions");
}*/

void Polygons::create_lod(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
				std::size_t target_vertices) {
	decimater.mesh() = *m_meshData;
	std::size_t targetDecimations = m_meshData->n_vertices() - target_vertices;
	std::size_t actualDecimations = decimater.decimate_to(target_vertices);
	// Flag the entire polygon as dirty
	// TODO: change number of triangles/quads!
	m_vertexAttributes.mark_changed(Device::CPU);
	logInfo("Decimated polygon mesh (", actualDecimations, "/", targetDecimations,
			" decimations performed");
	// TODO: this leaks mesh outside
}

} // namespace mufflon::scene::geometry