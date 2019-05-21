#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/textures/interface.hpp"
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
	m_matIndicesHdl(m_faceAttributes.add_attribute<u16>(MAT_INDICES_NAME)) {
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

Polygons::Polygons(const Polygons& poly) :
	m_meshData(std::make_unique<PolygonMeshType>(*poly.m_meshData)),
	m_vertexAttributes(*m_meshData),
	m_faceAttributes(*m_meshData),
	m_pointsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->points_pph())),
	m_normalsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec3f>(m_meshData->vertex_normals_pph())),
	m_uvsHdl(m_vertexAttributes.register_attribute<OpenMesh::Vec2f>(m_meshData->vertex_texcoords2D_pph())),
	m_matIndicesHdl(m_faceAttributes.add_attribute<u16>(MAT_INDICES_NAME)),
	m_indexFlags(poly.m_indexFlags),
	m_boundingBox(poly.m_boundingBox),
	m_triangles(poly.m_triangles),
	m_quads(poly.m_quads),
	m_uniqueMaterials(poly.m_uniqueMaterials) {
	m_vertexAttributes.copy(poly.m_vertexAttributes);
	m_faceAttributes.copy(poly.m_faceAttributes);
	// Copy the index and attribute buffers
	poly.m_indexBuffer.for_each([&](const auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& idxBuffer = m_indexBuffer.template get<ChangedBuffer>();
		idxBuffer.reserved = buffer.reserved;
		if(buffer.reserved == 0u || buffer.indices == ArrayDevHandle_t<ChangedBuffer::DEVICE, u32>{}) {
			idxBuffer.indices = ArrayDevHandle_t<ChangedBuffer::DEVICE, u32>{};
		} else {
			idxBuffer.indices = Allocator<ChangedBuffer::DEVICE>::template alloc_array<u32>(buffer.reserved);
			copy(idxBuffer.indices, buffer.indices, sizeof(u32) * buffer.reserved);
		}
	});
	poly.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& attribBuffer = m_attribBuffer.template get<ChangedBuffer>();
		attribBuffer.vertSize = buffer.vertSize;
		attribBuffer.faceSize = buffer.faceSize;
		if(buffer.vertSize == 0u || buffer.vertex == ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{}) {
			attribBuffer.vertex = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
		} else {
			attribBuffer.vertex = Allocator<ChangedBuffer::DEVICE>::template alloc_array<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.vertSize);
			copy(attribBuffer.vertex, buffer.vertex, sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.vertSize);
		}
		if(buffer.faceSize == 0u || buffer.face == ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{}) {
			attribBuffer.face = ArrayDevHandle_t<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>{};
		} else {
			attribBuffer.face = Allocator<ChangedBuffer::DEVICE>::template alloc_array<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.faceSize);
			copy(attribBuffer.face, buffer.face, sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.faceSize);
		}
	});
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
	m_quads(poly.m_quads),
	m_uniqueMaterials(std::move(poly.m_uniqueMaterials)) {
	// Move the index and attribute buffers
	poly.m_indexBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		m_indexBuffer.template get<ChangedBuffer>() = buffer;
		buffer.reserved = 0u;
		buffer.indices = ArrayDevHandle_t<ChangedBuffer::DEVICE, u32>{};
	});
	poly.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		m_attribBuffer.template get<ChangedBuffer>() = buffer;
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
		for(std::size_t i = startFace.idx(); i < startFace.idx() + numRead; ++i)
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

	return { hdl, readPoints, readNormals, readUvs };
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
	this->rebuild_index_buffer();
	logInfo("Tessellated polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

void Polygons::displace(tessellation::Tessellater& tessellater, const Scenario& scenario) {
	// Then perform tessellation
	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;
	tessellater.tessellate(*m_meshData);
	
	// Actual displacement: go over all vertices, check if one of its faces is displacement mapped
	// according to the provided material assignment and if yes, adjust the vertex position
	// along the normal (TODO: geometric or shading); if multiple adjacent faces disagree
	// about the displacement (e.g. two different displacement maps), then the average
	// is taken
	const auto* materialIndices = this->template acquire_const<Device::CPU, MaterialIndex>(get_material_indices_hdl());
	const auto* uvCoordinates = this->template acquire_const<Device::CPU, ei::Vec2>(get_uvs_hdl());
	auto* normals = this->template acquire<Device::CPU, ei::Vec3>(get_normals_hdl());
	auto* points = this->template acquire<Device::CPU, ei::Vec3>(get_points_hdl());
#pragma PARALLEL_FOR
	for(i64 i = 0; i < static_cast<i64>(m_meshData->n_vertices()); ++i) {
		const auto vertex = m_meshData->vertex_handle(static_cast<u32>(i));
		float displacement = 0.f;
		u32 faceCount = 0u;
		const ei::Vec2& uv = uvCoordinates[vertex.idx()];
		for(auto iter = m_meshData->cvf_ccwbegin(vertex); iter.is_valid(); ++iter) {
			const auto& mat = scenario.get_assigned_material(materialIndices[iter->idx()]);
			if(const auto dispMap = mat->get_displacement_map(); dispMap != nullptr) {
				const auto dispMapHdl = dispMap->acquire_const<Device::CPU>();

				displacement += mat->get_displacement_bias() + textures::sample(dispMapHdl, uv).x * mat->get_displacement_scale();
			}
			++faceCount;
		}

		if(displacement != 0.f) {
			const ei::Vec3& normal = normals[vertex.idx()];
			displacement /= static_cast<float>(faceCount);
			points[vertex.idx()] += normal * displacement;
			// TODO: compute new normal!
		}
	}

	// TODO: we recompute the geometric normals here, but we could probably compute them directly...
#pragma PARALLEL_FOR
	for(i64 i = 0; i < static_cast<i64>(m_meshData->n_vertices()); ++i) {
		const auto vertex = m_meshData->vertex_handle(static_cast<u32>(i));
		typename PolygonMeshType::Normal normal;
#pragma warning(push)
#pragma warning(disable : 4244)
		m_meshData->calc_vertex_normal_correct(vertex, normal);
#pragma warning(pop)
		normals[vertex.idx()] = ei::normalize(util::pun<ei::Vec3>(normal));
	}

	this->mark_changed(Device::CPU, get_points_hdl());
	this->mark_changed(Device::CPU, get_normals_hdl());
	this->rebuild_index_buffer();
	m_wasDisplaced = true;
	logInfo("Displacement mapped polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

// Creates a decimater 
OpenMesh::Decimater::DecimaterT<PolygonMeshType> Polygons::create_decimater() {
	return OpenMesh::Decimater::DecimaterT<PolygonMeshType>(*m_meshData);
}

std::size_t Polygons::decimate(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
							   std::size_t targetVertices, bool garbageCollect) {
	decimater.initialize();
	const std::size_t targetDecimations = decimater.mesh().n_vertices() - targetVertices;
	const std::size_t actualDecimations = decimater.decimate_to(targetVertices);

	if(garbageCollect)
		this->garbage_collect();
	else
		this->rebuild_index_buffer();
	// Do not garbage-collect the mesh yet - only rebuild the index buffer

	// Adjust vertex and face attribute sizes
	m_vertexAttributes.resize(m_meshData->n_vertices());
	m_faceAttributes.resize(m_meshData->n_faces());

	m_vertexAttributes.mark_changed(Device::CPU);
	if(targetVertices == 0) {
		logInfo("Decimated polygon mesh (", actualDecimations, " decimations performed; ",
				decimater.mesh().n_vertices() - actualDecimations, " vertices remaining)");
	} else {
		logInfo("Decimated polygon mesh (", actualDecimations, "/", targetDecimations,
				" decimations performed; ", decimater.mesh().n_vertices() - actualDecimations, " vertices remaining)");
	}
	// TODO: this leaks mesh outside

	return actualDecimations;
}

void Polygons::garbage_collect() {
	m_meshData->garbage_collection();
	this->rebuild_index_buffer();
}

void Polygons::transform(const ei::Mat3x4& transMat, const ei::Vec3& scale) {
	if(this->get_vertex_count() == 0) return;
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
	// Transform mesh
	ei::Mat3x3 rotation(transMat);
	ei::Vec3 translation(transMat[3], transMat[7], transMat[11]);
	ei::Vec3* vertices = m_vertexAttributes.acquire<Device::CPU, ei::Vec3>(m_pointsHdl);
	for(size_t i = 0; i < this->get_vertex_count(); i++) {
		vertices[i] *= scale;
		vertices[i] = rotation * vertices[i];
		vertices[i] += translation;
		m_boundingBox.max = ei::max(vertices[i], m_boundingBox.max);
		m_boundingBox.min = ei::min(vertices[i], m_boundingBox.min);
	}
	m_vertexAttributes.mark_changed(Device::CPU, m_pointsHdl);
	// Transform normals
	ei::Vec3* normals = m_vertexAttributes.acquire<Device::CPU, ei::Vec3>(m_normalsHdl);
	for(size_t i = 0; i < this->get_vertex_count(); i++) { // one normal per vertex
		normals[i] /= scale;
		normals[i] = rotation * normals[i];
	}
	m_vertexAttributes.mark_changed(Device::CPU, m_normalsHdl);
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

bool Polygons::has_displacement_mapping(const Scenario& scenario) const noexcept {
	for(MaterialIndex matIdx : m_uniqueMaterials)
		if(scenario.get_assigned_material(matIdx)->get_displacement_map() != nullptr)
			return true;

	return false;
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

void Polygons::rebuild_index_buffer() {
	// Update the statistics we keep
	// TODO: is there a better way?
	m_triangles = 0u;
	m_quads = 0u;
	for(const auto& face : this->faces()) {
		const std::size_t vertices = std::distance(face.begin(), face.end());
		if(vertices == 0)
			continue;
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

	for(auto face : m_meshData->faces()) {
		const auto startVertex = m_meshData->cfv_ccwbegin(face);
		const auto endVertex = m_meshData->cfv_ccwend(face);
		const auto vertexCount = std::distance(startVertex, endVertex);

		u32* currIndices = indexBuffer;
		if(vertexCount == 3u) {
			currIndices += 3u * currTri++;
		} else {
			currIndices += 3u * m_triangles + 4u * currQuad++;
		}
		for(auto vertexIter = startVertex; vertexIter.is_valid(); ++vertexIter) {
			*(currIndices++) = static_cast<u32>(vertexIter->idx());
			ei::Vec3 pt = util::pun<ei::Vec3>(m_meshData->point(*vertexIter));
			m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ pt } };
		}
	}

	// Let our attribute pools know that we changed sizes
	m_vertexAttributes.resize(m_meshData->n_vertices());
	m_faceAttributes.resize(m_meshData->n_faces());

	// Flag the entire polygon as dirty
	m_vertexAttributes.mark_changed(Device::CPU);
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