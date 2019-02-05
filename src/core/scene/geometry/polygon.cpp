#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "core/scene/descriptors.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include "core/scene/tessellation/tessellater.hpp"

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
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
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

std::size_t Polygons::add_bulk(StringView name, const VertexHandle& startVertex,
					 std::size_t count, util::IByteReader& attrStream) {
	mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
	return m_vertexAttributes.restore(name, attrStream, static_cast<std::size_t>(startVertex.idx()), count);
}

std::size_t Polygons::add_bulk(StringView name, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	return this->add_bulk(m_faceAttributes.get_attribute_handle(name), startFace,
						  count, attrStream);
}

std::size_t Polygons::add_bulk(VertexAttributeHandle hdl, const VertexHandle& startVertex,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
	return m_vertexAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startVertex.idx()), count);
}

std::size_t Polygons::add_bulk(FaceAttributeHandle hdl, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData->n_vertices());
	std::size_t numRead = m_faceAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startFace.idx()), count);
	// Update material table in case this load was about materials
	if(hdl == m_matIndicesHdl) {
		MaterialIndex* materials = m_faceAttributes.acquire<Device::CPU, MaterialIndex>(hdl);
		for(std::size_t i = startFace.idx(); i < startFace.idx()+numRead; ++i)
			m_uniqueMaterials.emplace(materials[i]);
	}
	return numRead;
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
	m_faceAttributes.acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl.idx()] = idx;
	m_faceAttributes.mark_changed(Device::CPU, m_matIndicesHdl);
	m_uniqueMaterials.emplace(idx);
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
	m_faceAttributes.acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl.idx()] = idx;
	m_faceAttributes.mark_changed(Device::CPU, m_matIndicesHdl);
	m_uniqueMaterials.emplace(idx);
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
	std::size_t readNormals = m_vertexAttributes.restore(m_normalsHdl, normalStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
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
	std::size_t readNormals = m_vertexAttributes.restore(m_normalsHdl, normalStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
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
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
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
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
	// Expand the bounding box
	m_boundingBox.max = ei::max(boundingBox.max, m_boundingBox.max);
	m_boundingBox.min = ei::min(boundingBox.min, m_boundingBox.min);

	return { hdl, readPoints, 0u, readUvs };
}

void Polygons::tessellate(tessellation::Tessellater& tessellater) {

	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;
	tessellater.tessellate(*m_meshData);

	/*
	tessellater(*m_meshData, divisions);*/
	// TODO: change number of triangles/quads!

	// Let our attribute pools know that we changed sizes
	m_vertexAttributes.resize(m_meshData->n_vertices());
	m_faceAttributes.resize(m_meshData->n_faces());

	// Update the statistics we keep
	// TODO: is there a better way?
	m_triangles = 0u;
	m_quads = 0u;
	for(const auto& face : this->faces()) {
		const std::size_t vertices = std::distance(face.begin(), face.end());
		if(vertices == 3u)
			++m_triangles;
		else if(vertices == 4u)
			++m_quads;
		else
			throw std::runtime_error("Tessellation added a non-quad/tri face (" + std::to_string(vertices) + " vertices)");
	}

	// Invalidate bounding box
	m_boundingBox.min = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max()
	};
	m_boundingBox.max = {
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
	};

	// Rebuild the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * m_quads);
	std::size_t currTri = 0u;
	std::size_t currQuad = 0u;
	u32* indexBuffer = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices;
	for(const auto& face : this->faces()) {
		u32* currIndices = indexBuffer;
		if(std::distance(face.begin(), face.end()) == 3u) {
			currIndices += 3u * currTri++;
		} else {
			currIndices += 3u * m_triangles + 4u * currQuad++;
		}
		for(auto vertexIter = face.begin(); vertexIter != face.end(); ++vertexIter) {
			*(currIndices++) = static_cast<u32>(vertexIter->idx());
			ei::Vec3 pt = util::pun<ei::Vec3>(m_meshData->points()[vertexIter->idx()]);
			m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ util::pun<ei::Vec3>(m_meshData->points()[vertexIter->idx()]) } };
		}
	}

	// Flag the entire polygon as dirty
	m_vertexAttributes.mark_changed(Device::CPU);
	logInfo("Uniformly tessellated polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
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

template < Device dev >
void Polygons::synchronize() {
	m_vertexAttributes.synchronize<dev>();
	m_faceAttributes.synchronize<dev>();
	// Synchronize the index buffer
	if(m_indexFlags.needs_sync(dev) && m_indexFlags.has_changes()) {
		if(m_indexFlags.has_competing_changes()) {
			logError("[Polygons::synchronize] Competing device changes; ignoring one");
		}
		m_indexBuffer.for_each([&](auto& buffer) {
			using ChangedBuffer = std::decay_t<decltype(buffer)>;
			this->synchronize_index_buffer<ChangedBuffer::DEVICE, dev>();
		});
	}
}

template < Device dev >
PolygonsDescriptor<dev> Polygons::get_descriptor() {
	this->synchronize<dev>();
	return PolygonsDescriptor<dev>{
		static_cast<u32>(this->get_vertex_count()),
		static_cast<u32>(this->get_triangle_count()),
		static_cast<u32>(this->get_quad_count()),
		0u,
		0u,
		this->acquire_const<dev, ei::Vec3>(this->get_points_hdl()),
		this->acquire_const<dev, ei::Vec3>(this->get_normals_hdl()),
		this->acquire_const<dev, ei::Vec2>(this->get_uvs_hdl()),
		this->acquire_const<dev, u16>(this->get_material_indices_hdl()),
		this->get_index_buffer<dev>(),
		ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{},
		ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{}
	};
}

template < Device dev >
void Polygons::update_attribute_descriptor(PolygonsDescriptor<dev>& descriptor,
										   const std::vector<const char*>& vertexAttribs,
										   const std::vector<const char*>& faceAttribs) {
	this->synchronize<dev>();
	// Free the previous attribute array if no attributes are wanted
	auto& buffer = m_attribBuffer.template get<AttribBuffer<dev>>();

	if(vertexAttribs.size() == 0 && buffer.vertSize != 0) {
		buffer.vertex = Allocator<dev>::free(buffer.vertex, buffer.vertSize);
	} else {
		if(buffer.vertSize == 0)
			buffer.vertex = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(vertexAttribs.size());
		else
			buffer.vertex = Allocator<dev>::realloc(buffer.vertex, buffer.vertSize, vertexAttribs.size());
	}
	if(faceAttribs.size() == 0 && buffer.faceSize != 0) {
		buffer.face = Allocator<dev>::free(buffer.face, buffer.faceSize);
	} else {
		if(buffer.faceSize == 0)
			buffer.face = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(faceAttribs.size());
		else
			buffer.face = Allocator<dev>::realloc(buffer.face, buffer.faceSize, faceAttribs.size());
	}

	std::vector<void*> cpuVertexAttribs(vertexAttribs.size());
	std::vector<void*> cpuFaceAttribs(faceAttribs.size());
	for(const char* name : vertexAttribs)
		cpuVertexAttribs.push_back(m_vertexAttributes.acquire<dev, void>(name));
	for(const char* name : faceAttribs)
		cpuFaceAttribs.push_back(m_faceAttributes.acquire<dev, void>(name));
	copy<void*>(buffer.vertex, cpuVertexAttribs.data(), sizeof(void*) * vertexAttribs.size());
	copy<void*>(buffer.face, cpuFaceAttribs.data(), sizeof(void*) *faceAttribs.size());

	descriptor.numVertexAttributes = static_cast<u32>(vertexAttribs.size());
	descriptor.numFaceAttributes = static_cast<u32>(faceAttribs.size());
	descriptor.vertexAttributes = buffer.vertex;
	descriptor.faceAttributes = buffer.face;
}

// Reserves more space for the index buffer
template < Device dev >
void Polygons::reserve_index_buffer(std::size_t capacity) {
	auto& buffer = m_indexBuffer.get<IndexBuffer<dev>>();
	if(capacity > buffer.reserved) {
		if(buffer.reserved == 0u)
			buffer.indices = Allocator<dev>::template alloc_array<u32>(capacity);
		else
			buffer.indices = Allocator<Device::CPU>::realloc(buffer.indices, buffer.reserved,
															 capacity);
		buffer.reserved = capacity;
		m_indexFlags.mark_changed(dev);
	}
}

// Synchronizes two device index buffers
template < Device changed, Device sync >
void Polygons::synchronize_index_buffer() {
	if constexpr(changed != sync) {
		if(m_indexFlags.has_changes(changed)) {
			auto& changedBuffer = m_indexBuffer.get<IndexBuffer<changed>>();
			auto& syncBuffer = m_indexBuffer.get<IndexBuffer<sync>>();

			// Check if we need to realloc
			if(syncBuffer.reserved < m_triangles + m_quads)
				this->reserve_index_buffer<sync>(3u * m_triangles + 4u * m_quads);

			if(changedBuffer.reserved != 0u)
				copy(syncBuffer.indices, changedBuffer.indices, sizeof(u32) * (3u * m_triangles + 4u * m_quads));
			m_indexFlags.mark_synced(sync);
		}
	}
}

template < Device dev >
void Polygons::resizeAttribBuffer(std::size_t v, std::size_t f) {
	AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
	// Resize the attribute array if necessary
	if(attribBuffer.faceSize < f) {
		if(attribBuffer.faceSize == 0)
			attribBuffer.face = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(f);
		else
			attribBuffer.face = Allocator<dev>::realloc(attribBuffer.face, attribBuffer.faceSize, f);
		attribBuffer.faceSize = f;
	}
	if(attribBuffer.vertSize < v) {
		if(attribBuffer.vertSize == 0)
			attribBuffer.vertex = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(v);
		else
			attribBuffer.vertex = Allocator<dev>::realloc(attribBuffer.vertex, attribBuffer.vertSize, v);
		attribBuffer.vertSize = v;
	}
}

// Explicit instantiations
template void Polygons::reserve_index_buffer<Device::CPU>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::CUDA>(std::size_t capacity);
//template void Polygons::reserve_index_buffer<Device::OPENGL>(std::size_t capacity);
template void Polygons::synchronize_index_buffer<Device::CPU, Device::CUDA>();
//template void Polygons::synchronize_index_buffer<Device::CPU, Device::OPENGL>();
template void Polygons::synchronize_index_buffer<Device::CUDA, Device::CPU>();
//template void Polygons::synchronize_index_buffer<Device::CUDA, Device::OPENGL>();
//template void Polygons::synchronize_index_buffer<Device::OPENGL, Device::CPU>();
//template void Polygons::synchronize_index_buffer<Device::OPENGL, Device::CUDA>();
template void Polygons::resizeAttribBuffer<Device::CPU>(std::size_t v, std::size_t f);
template void Polygons::resizeAttribBuffer<Device::CUDA>(std::size_t v, std::size_t f);
//template void Polygons::resizeAttribBuffer<Device::OPENGL>(std::size_t v, std::size_t f);
template void Polygons::synchronize<Device::CPU>();
template void Polygons::synchronize<Device::CUDA>();
//template void Polygons::synchronize<Device::OPENGL>();
template PolygonsDescriptor<Device::CPU> Polygons::get_descriptor<Device::CPU>();
template PolygonsDescriptor<Device::CUDA> Polygons::get_descriptor<Device::CUDA>();
template void Polygons::update_attribute_descriptor<Device::CPU>(PolygonsDescriptor<Device::CPU>& descriptor,
																  const std::vector<const char*>& vertexAttribs,
																  const std::vector<const char*>& faceAttribs);
template void Polygons::update_attribute_descriptor<Device::CUDA>(PolygonsDescriptor<Device::CUDA>& descriptor,
																  const std::vector<const char*>& vertexAttribs,
																  const std::vector<const char*>& faceAttribs);
/*template PolygonsDescriptor<Device::OPENGL> Polygons::get_descriptor<Device::OPENGL>(const std::vector<const char*>& vertexAttribs,
																					 const std::vector<const char*>& faceAttribs);*/


} // namespace mufflon::scene::geometry