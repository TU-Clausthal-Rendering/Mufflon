
#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/data_structs/count_octree.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/clustering/util.hpp"
#include "core/math/curvature.hpp"
#include "core/scene/tessellation/tessellater.hpp"
#include "core/scene/tessellation/displacement_mapper.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <cmath>

namespace mufflon::scene::geometry {

// Default construction, creates material-index attribute.
Polygons::Polygons() :
	m_vertexAttributes(),
	m_faceAttributes(),
	m_pointsHdl{ this->template add_vertex_attribute<ei::Vec3>("points") },
	m_normalsHdl{ this->template add_vertex_attribute<ei::Vec3>("normals") },
	m_uvsHdl{ this->template add_vertex_attribute<ei::Vec2>("uvs") },
	m_curvRefCount{ 0u },
	m_curvatureHdl{},
	m_animationWeightHdl{},
	m_matIndicesHdl{ this->template add_face_attribute<MaterialIndex>("materials") },
	m_indexBuffer{},
	m_attribBuffer{},
	m_descFlags{},
	m_triangles{ 0u },
	m_quads{ 0u },
	m_wasDisplaced{ false }
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

Polygons::Polygons(const PolygonMeshType& mesh, const OpenMesh::FPropHandleT<MaterialIndex> mats) :
	Polygons{}
{
	m_vertexAttributes.resize(mesh.n_vertices());
	m_faceAttributes.resize(mesh.n_faces());
	auto* points = this->template acquire<Device::CPU, ei::Vec3>(this->get_points_hdl());
	auto* normals = this->template acquire<Device::CPU, ei::Vec3>(this->get_normals_hdl());
	auto* uvs = this->template acquire<Device::CPU, ei::Vec2>(this->get_uvs_hdl());
	for(const auto vertex : mesh.vertices()) {
		const auto point = util::pun<ei::Vec3>(mesh.point(vertex));
		points[vertex.idx()] = point;
		normals[vertex.idx()] = util::pun<ei::Vec3>(mesh.normal(vertex));
		uvs[vertex.idx()] = util::pun<ei::Vec2>(mesh.texcoord2D(vertex));
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ point } };
	}

	auto* matIndices = this->template acquire<Device::CPU, MaterialIndex>(this->get_material_indices_hdl());
	for(const auto face : mesh.faces()) {
		if(mats.is_valid())
			matIndices[face.idx()] = mesh.property(mats, face);
		else
			matIndices[face.idx()] = 0u;

		if(std::distance(mesh.cfv_ccwbegin(face), mesh.cfv_ccwend(face)) == 3u)
			++m_triangles;
		else
			++m_quads;
	}

	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * m_quads);
	auto* indices = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices.get();
	auto* currTri = indices;
	auto* currQuad = indices + 3u * m_triangles;
	for(const auto face : mesh.faces()) {
		u32** ind;
		if(std::distance(mesh.cfv_ccwbegin(face), mesh.cfv_ccwend(face)) == 3u)
			ind = &currTri;
		else
			ind = &currQuad;

		for(auto iter = mesh.cfv_ccwbegin(face); iter != mesh.cfv_ccwend(face); ++iter) {
			**ind = iter->idx();
			(*ind) += 1u;
		}
	}
}

Polygons::Polygons(const Polygons& poly) :
	m_vertexAttributes{ poly.m_vertexAttributes },
	m_faceAttributes{ poly.m_faceAttributes },
	m_pointsHdl{ poly.m_pointsHdl },
	m_normalsHdl{ poly.m_normalsHdl },
	m_uvsHdl{ poly.m_uvsHdl },
	m_curvatureHdl{ poly.m_curvatureHdl },
	m_matIndicesHdl{ poly.m_matIndicesHdl },
	m_boundingBox{ poly.m_boundingBox },
	m_triangles{ poly.m_triangles },
	m_quads{ poly.m_quads }
{
	// Copy the index and attribute buffers
	poly.m_indexBuffer.for_each([&](const auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& idxBuffer = m_indexBuffer.template get<ChangedBuffer>();
		idxBuffer.reserved = buffer.reserved;
		if(buffer.reserved == 0u || buffer.indices.get() == nullptr) {
			idxBuffer.indices.reset();
		} else {
			idxBuffer.indices = make_udevptr_array<ChangedBuffer::DEVICE, u32, false>(buffer.reserved);
			copy(idxBuffer.indices.get(), buffer.indices.get(), sizeof(u32) * buffer.reserved);
		}
	});
	poly.m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		auto& attribBuffer = m_attribBuffer.template get<ChangedBuffer>();
		attribBuffer.vertSize = buffer.vertSize;
		attribBuffer.faceSize = buffer.faceSize;
		if(buffer.vertSize == 0u || buffer.vertex.get() == nullptr) {
			attribBuffer.vertex.reset();
		} else {
			attribBuffer.vertex = make_udevptr_array<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>, false>(buffer.vertSize);
			copy(attribBuffer.vertex.get(), buffer.vertex.get(), sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.vertSize);
		}
		if(buffer.faceSize == 0u || buffer.face.get() == nullptr) {
			attribBuffer.face.reset();
		} else {
			attribBuffer.face = make_udevptr_array<ChangedBuffer::DEVICE, ArrayDevHandle_t<ChangedBuffer::DEVICE, void>, false>(buffer.faceSize);
			copy(attribBuffer.face.get(), buffer.face.get(), sizeof(ArrayDevHandle_t<ChangedBuffer::DEVICE, void>) * buffer.faceSize);
		}
	});
}

Polygons::Polygons(Polygons&& poly) :
	m_vertexAttributes{ std::move(poly.m_vertexAttributes) },
	m_faceAttributes{ std::move(poly.m_faceAttributes) },
	m_pointsHdl{ poly.m_pointsHdl },
	m_normalsHdl{ poly.m_normalsHdl },
	m_uvsHdl{ poly.m_uvsHdl },
	m_curvatureHdl{ poly.m_curvatureHdl },
	m_matIndicesHdl{ poly.m_matIndicesHdl },
	m_indexBuffer{ std::move(poly.m_indexBuffer) },
	m_attribBuffer{ std::move(poly.m_attribBuffer) },
	m_boundingBox{ poly.m_boundingBox },
	m_triangles{ poly.m_triangles },
	m_quads{ poly.m_quads }
{}

std::size_t Polygons::add_bulk(const VertexAttributeHandle& hdl, const VertexHandle& startVertex,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(static_cast<std::size_t>(startVertex) < this->get_vertex_count());
	return m_vertexAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startVertex), count);
}

std::size_t Polygons::add_bulk(const FaceAttributeHandle& hdl, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(static_cast<std::size_t>(startFace) < this->get_face_count());
	return m_faceAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startFace), count);
}


void Polygons::reserve(std::size_t vertices, std::size_t tris, std::size_t quads) {
	m_vertexAttributes.reserve(vertices);
	m_faceAttributes.reserve(tris + quads);
	this->reserve_index_buffer<Device::CPU>(3u * tris + 4u * quads);
}

Polygons::VertexHandle Polygons::add(const Point& point, const Normal& normal,
									 const UvCoordinate& uv) {
	const auto vh = static_cast<VertexHandle>(this->get_vertex_count());
	// Resize the attribute and set the vertex data
	m_vertexAttributes.resize(m_vertexAttributes.get_attribute_elem_count() + 1u);

	m_vertexAttributes.template acquire<Device::CPU, ei::Vec3>(m_pointsHdl)[vh] = point;
	m_vertexAttributes.template acquire<Device::CPU, ei::Vec3>(m_normalsHdl)[vh] = normal;
	m_vertexAttributes.template acquire<Device::CPU, ei::Vec2>(m_uvsHdl)[vh] = uv;
	m_vertexAttributes.mark_changed(Device::CPU);
	// Expand the mesh's bounding box
	m_boundingBox = ei::Box(m_boundingBox, ei::Box(point));

	return vh;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2) {
	mAssert(static_cast<std::size_t>(v0) < this->get_vertex_count());
	mAssert(static_cast<std::size_t>(v1) < this->get_vertex_count());
	mAssert(static_cast<std::size_t>(v2) < this->get_vertex_count());
	mAssert(m_quads == 0u); // To keep the order implicitly
	const auto hdl = static_cast<FaceHandle>(this->get_face_count());
	// Expand the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * (m_triangles + 1u));
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices.get();
	std::size_t currIndexCount = 3u * m_triangles;
	indexBuffer[currIndexCount + 0u] = v0;
	indexBuffer[currIndexCount + 1u] = v1;
	indexBuffer[currIndexCount + 2u] = v2;

	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_triangles;
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2, MaterialIndex idx) {
	TriangleHandle hdl = this->add(v0, v1, v2);
	m_faceAttributes.template acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl] = idx;
	m_faceAttributes.mark_changed(Device::CPU);
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const Triangle& tri) {
	return this->add(VertexHandle(tri[0u]), VertexHandle(tri[1u]), VertexHandle(tri[2u]));
}

Polygons::TriangleHandle Polygons::add(const Triangle& tri, MaterialIndex idx) {
	return this->add(VertexHandle(tri[0u]), VertexHandle(tri[1u]),
					 VertexHandle(tri[2u]), idx);
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3) {
	mAssert(static_cast<std::size_t>(v0) < this->get_vertex_count());
	mAssert(static_cast<std::size_t>(v1) < this->get_vertex_count());
	mAssert(static_cast<std::size_t>(v2) < this->get_vertex_count());
	mAssert(static_cast<std::size_t>(v3) < this->get_vertex_count());
	const auto hdl = static_cast<FaceHandle>(this->get_face_count());
	// Expand the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * (m_quads + 1u));
	auto indexBuffer = m_indexBuffer.get<IndexBuffer<Device::CPU>>().indices.get();
	std::size_t currIndexCount = 3u * m_triangles + 4u * m_quads;
	indexBuffer[currIndexCount + 0u] = v0;
	indexBuffer[currIndexCount + 1u] = v1;
	indexBuffer[currIndexCount + 2u] = v2;
	indexBuffer[currIndexCount + 3u] = v3;
	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_quads;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3,
								   MaterialIndex idx) {
	QuadHandle hdl = this->add(v0, v1, v2, v3);
	m_faceAttributes.template acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl] = idx;
	m_faceAttributes.mark_changed(Device::CPU);
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

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& normalStream, util::IByteReader& uvStream) {
	const auto start = this->get_vertex_count();
	mAssert(start < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	const auto hdl = static_cast<VertexHandle>(start);

	// Resize the attributes prior
	this->reserve(start + count, m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readNormals = m_vertexAttributes.restore(m_normalsHdl, normalStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
	// Expand the bounding box
	const auto* points = m_vertexAttributes.template acquire_const<Device::CPU, ei::Vec3>(m_pointsHdl);
	m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ points, static_cast<u32>(readPoints) } };
	return { hdl, readPoints, readNormals, readUvs };
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& normalStream, util::IByteReader& uvStream,
											  const ei::Box& boundingBox) {
	const auto start = this->get_vertex_count();
	mAssert(start < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	const auto hdl = static_cast<VertexHandle>(start);

	// Resize the attributes prior
	this->reserve(start + count, m_triangles, m_quads);

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
	const auto start = this->get_vertex_count();
	mAssert(start < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	const auto hdl = static_cast<VertexHandle>(start);

	// Resize the attributes prior
	this->reserve(start + count, m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
	// Expand the bounding box
	const OpenMesh::Vec3f* points = m_vertexAttributes.template acquire_const<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl);
	for(std::size_t i = start; i < start + readPoints; ++i) {
		m_boundingBox.max = ei::max(util::pun<ei::Vec3>(points[i]), m_boundingBox.max);
		m_boundingBox.min = ei::min(util::pun<ei::Vec3>(points[i]), m_boundingBox.min);
	}

	return { hdl, readPoints, 0u, readUvs };
}

Polygons::VertexBulkReturn Polygons::add_bulk(std::size_t count, util::IByteReader& pointStream,
											  util::IByteReader& uvStream, const ei::Box& boundingBox) {
	const auto start = this->get_vertex_count();
	mAssert(start < static_cast<std::size_t>(std::numeric_limits<int>::max()));
	const auto hdl = static_cast<VertexHandle>(start);

	// Resize the attributes prior
	this->reserve(start + count, m_triangles, m_quads);

	// Read the attributes
	std::size_t readPoints = m_vertexAttributes.restore(m_pointsHdl, pointStream, start, count);
	std::size_t readUvs = m_vertexAttributes.restore(m_uvsHdl, uvStream, start, count);
	// Expand the bounding box
	m_boundingBox.max = ei::max(boundingBox.max, m_boundingBox.max);
	m_boundingBox.min = ei::min(boundingBox.min, m_boundingBox.min);

	return { hdl, readPoints, 0u, readUvs };
}


void Polygons::create_halfedge_structure(PolygonMeshType& mesh) {
	this->template synchronize<Device::CPU>();
	const auto vertexCount = this->get_vertex_count();
	const auto faceCount = this->get_face_count();
	// Estimate the number of edges we need with Euler characteristic for stellated dodecahedron
	mesh.reserve(vertexCount, vertexCount + faceCount + 6, faceCount);
	// Add all vertices
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	const auto* normals = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_normals_hdl());
	const auto* uvs = this->template acquire_const<Device::CPU, ei::Vec2>(this->get_uvs_hdl());
	for(std::size_t vertex = 0u; vertex < vertexCount; ++vertex) {
		const auto hdl = mesh.add_vertex(util::pun<OpenMesh::Vec3f>(points[vertex]));
		mesh.set_normal(hdl, util::pun<OpenMesh::Vec3f>(normals[vertex]));
		mesh.set_texcoord2D(hdl, util::pun<OpenMesh::Vec2f>(uvs[vertex]));
	}
	// Add all faces
	const auto* indices = this->get_index_buffer<Device::CPU>();
	for(std::size_t tri = 0u; tri < m_triangles; ++tri) {
		const auto faceHdl = mesh.add_face(
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * tri + 0u]) },
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * tri + 1u]) },
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * tri + 2u]) }
		);
	}
	for(std::size_t quad = 0u; quad < m_quads; ++quad) {
		const auto faceHdl = mesh.add_face(
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * m_triangles + 4u * quad + 0u]) },
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * m_triangles + 4u * quad + 1u]) },
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * m_triangles + 4u * quad + 2u]) },
			OpenMesh::VertexHandle{ static_cast<int>(indices[3u * m_triangles + 4u * quad + 3u]) }
		);
	}
}

PolygonMeshType Polygons::create_halfedge_structure() {
	PolygonMeshType mesh;
	this->create_halfedge_structure(mesh);
	return mesh;
}

void Polygons::reconstruct_from_reduced_mesh(const PolygonMeshType& mesh, std::vector<u32>* newVertexPosition,
											 std::vector<ei::Vec3>* normals) {
	// We have to keep track of moved vertices. It would be beneficial memory-wise
	// to use a map for moved ones, but faster to just use a linear array
	std::vector<u32> buffer;
	if(newVertexPosition == nullptr)
		newVertexPosition = &buffer;
	newVertexPosition->resize(mesh.n_vertices());
	std::fill(newVertexPosition->begin(), newVertexPosition->end(), std::numeric_limits<u32>::max());

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

	auto currPos = 0u;
	auto currEnd = mesh.n_vertices() - 1u;
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	// Find the next vertex from the back to move to a potential hole
	while(currEnd > currPos && mesh.status(mesh.vertex_handle(static_cast<unsigned>(currEnd))).deleted())
		currEnd -= 1u;
	// It is important that the iterator does not skip deleted vertices (we do that ourselves)!
	for(auto iter = mesh.vertices_begin(); iter != mesh.vertices_end(); ++iter) {
		if(currPos >= currEnd) {
			// Increment the position (and thus the size) if our last vertex is not deleted
			if(!mesh.status(*iter).deleted())
				++currPos;
			break;
		}

		const auto vertex = *iter;
		// If the vertex is deleted, then we take the last non-deleted vertex to fill the gap
		if(mesh.status(vertex).deleted()) {
			m_vertexAttributes.copy(currEnd, currPos);
			(*newVertexPosition)[currEnd] = static_cast<u32>(currPos);

			// Find the next vertex from the back to move to a potential hole
			do {
				currEnd -= 1u;
			} while(currEnd > currPos && mesh.status(mesh.vertex_handle(static_cast<unsigned>(currEnd))).deleted());
		}

		// Expand bounding box
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ points[currPos] } };
		currPos += 1u;
	}
	m_vertexAttributes.resize(currPos);
	m_vertexAttributes.shrink_to_fit();

	// Reconstruct the index buffer - count triangles and quads first for correct offsets
	if(m_triangles == 0u && m_quads == 0u)
		throw std::runtime_error("Something went wrong: no faces!");
	m_triangles = 0u;
	m_quads = 0u;
	for(const auto face : mesh.faces()) {
		const auto vBegin = mesh.cfv_ccwbegin(face);
		const auto vEnd = mesh.cfv_ccwend(face);
		const auto vertexCount = std::distance(vBegin, vEnd);
		if(vertexCount == 0)
			continue;
		if(vertexCount == 3u)
			++m_triangles;
		else if(vertexCount == 4u)
			++m_quads;
		else
			throw std::runtime_error("Found a non-quad/tri face (" + std::to_string(vertexCount) + " vertices)");
	}
	if(m_triangles == 0u && m_quads == 0u)
		throw std::runtime_error("Something went wrong: no faces!");

	// With the correct tri/quad counts we can resize the index buffer
	this->unload_index_buffer<Device::CPU>();
	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * m_quads);
	auto* indexBuffer = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices.get();

	// We don't bother with switching around faces and whatnot - we simply allocate a new buffer
	// TODO: that DOES cost a lot of extra memory, maybe we should switch around
	FaceAttributePoolType faceAttribs{ m_faceAttributes };
	std::size_t currTri = 0u;
	std::size_t currQuad = 0u;
	for(const auto face : mesh.faces()) {
		const auto startVertex = mesh.cfv_ccwbegin(face);
		const auto endVertex = mesh.cfv_ccwend(face);
		const auto vertexCount = std::distance(startVertex, endVertex);

		u32* currIndices = indexBuffer;
		std::size_t faceIndex;
		if(vertexCount == 3u) {
			faceIndex = currTri;
			currIndices += 3u * currTri++;
		} else {
			faceIndex = m_triangles + currQuad;
			currIndices += 3u * m_triangles + 4u * currQuad++;
		}

		// Copy the attributes (we can make the assumption of face.idx() == index in face attributes
		// because the mesh was created by first inserting triangles, then quads, and no faces were
		// garbage collected yet)
		faceAttribs.copy(m_faceAttributes, static_cast<std::size_t>(face.idx()), faceIndex);
		// Set the index buffer
		for(auto vertexIter = startVertex; vertexIter != endVertex; ++vertexIter) {
			const auto vertexIdx = static_cast<u32>(vertexIter->idx());
			auto index = (*newVertexPosition)[vertexIdx];
			if(index == std::numeric_limits<u32>::max())
				index = vertexIdx;
			mAssert(index < m_vertexAttributes.get_attribute_elem_count());
			*(currIndices++) = index;
		}
	}



	m_faceAttributes = std::move(faceAttribs);
	this->mark_changed(Device::CPU);

	// To recalculate the normals we would either have to garbage collect the mesh,
	// use skipping iterators (severe overhead in heavily decimated meshes), or use
	// a buffer to keep track of cumulative normals
	recompute_vertex_normals(normals);
}

void Polygons::recompute_vertex_normals(std::vector<ei::Vec3>* normals) {
	// TODO: this really lends itself to SIMD
	std::vector<ei::Vec3> normalBuffer;
	if(normals == nullptr)
		normals = &normalBuffer;
	normals->resize(m_vertexAttributes.get_attribute_elem_count());
	std::fill(normals->begin(), normals->end(), ei::Vec3{ 0.f });
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());

	for(const auto tri : this->triangles()) {
		// Compute the face normal
		ei::Vec3 edges[3u];
		{
			ei::Vec3 triPoints[3u];
			for(unsigned i = 0u; i < 3u; ++i)
				triPoints[i] = points[tri[i]];
			for(unsigned i = 0u; i < 3u; ++i)
				edges[i] = triPoints[(i + 1u) % 3u] - triPoints[i];
		}
		const auto normal = ei::cross(edges[0], edges[1]);

		for(unsigned i = 0u; i < 3u; ++i) {
			const auto angle = std::acos(-ei::dot(edges[(i + 3u) % 3u], edges[i]));
			(*normals)[tri[i]] += normal * angle;
		}
	}

	for(const auto quad : this->quads()) {
		// Compute the face normals
		ei::Vec3 edges[4u];
		{
			ei::Vec3 quadPoints[4u];
			for(unsigned i = 0u; i < 4u; ++i)
				quadPoints[i] = points[quad[i]];
			for(unsigned i = 0u; i < 4u; ++i)
				edges[i] = quadPoints[(i + 1u) % 4u] - quadPoints[i];
		}

		for(unsigned i = 0u; i < 4u; ++i) {
			const auto normal = ei::cross(edges[(i + 4u) % 4u], edges[i]);
			const auto angle = std::acos(-ei::dot(edges[(i + 4u) % 4u], edges[i]));
			(*normals)[quad[i]] += normal * angle;
		}
	}

	// Normalize and write to the buffer
	auto* vertexNormals = this->template acquire<Device::CPU, ei::Vec3>(this->get_normals_hdl());
	for(std::size_t i = 0u; i < normals->size(); ++i) {
		vertexNormals[i] = ei::normalize((*normals)[i]);
	}
}

#if 0
void Polygons::cluster_uniformly(const ei::UVec3& gridRes) {
	// Idea taken from here: https://www.comp.nus.edu.sg/~tants/Paper/simplify.pdf

	this->template synchronize<Device::CPU>();
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	// Gather all edges and compute the error quadrics per vertex
	// We'll have lots of duplicates, but that's (imo) fine and worth the reduced
	// runtime overhead
	const auto compute_quadric = [points](const ei::UVec3& indices) {
		// Quadric
		const auto p0 = points[indices.x];
		auto normal = ei::cross(points[indices.y] - p0, points[indices.z] - p0);
		auto area = ei::len(normal);
		if(area > std::numeric_limits<decltype(area)>::min()) {
			normal /= area;
			area *= 0.5f;
		}
		const auto d = -ei::dot(p0, normal);
		OpenMesh::Geometry::Quadricf q{ normal.x, normal.y, normal.z, d };
		q *= area;
		return q;
	};

	std::vector<std::pair<u32, u32>> edges;
	std::vector<OpenMesh::Geometry::Quadricf> quadrics(this->get_vertex_count());
	edges.reserve(m_triangles * 6u + m_quads * 8u);
	for(const auto tri : this->triangles()) {
		edges.emplace_back(tri.x, tri.y);
		edges.emplace_back(tri.y, tri.z);
		edges.emplace_back(tri.z, tri.x);
		edges.emplace_back(tri.x, tri.z);
		edges.emplace_back(tri.z, tri.y);
		edges.emplace_back(tri.y, tri.x);

		const auto q = compute_quadric(tri);
		quadrics[tri.x] += q;
		quadrics[tri.y] += q;
		quadrics[tri.z] += q;
	}
	for(const auto quad : this->quads()) {
		edges.emplace_back(quad.x, quad.y);
		edges.emplace_back(quad.y, quad.z);
		edges.emplace_back(quad.z, quad.w);
		edges.emplace_back(quad.w, quad.x);
		edges.emplace_back(quad.x, quad.w);
		edges.emplace_back(quad.w, quad.z);
		edges.emplace_back(quad.z, quad.y);
		edges.emplace_back(quad.y, quad.x);

		const ei::UVec3 tri0{ quad.x, quad.y, quad.z };
		const ei::UVec3 tri1{ quad.x, quad.z, quad.w };
		const auto q0 = compute_quadric(tri0);
		const auto q1 = compute_quadric(tri1);
		quadrics[tri0.x] += q0;
		quadrics[tri0.y] += q0;
		quadrics[tri0.z] += q0;
		quadrics[tri1.x] += q1;
		quadrics[tri1.y] += q1;
		quadrics[tri1.z] += q1;
	}
	// Sort them by vertex
	std::sort(edges.begin(), edges.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
	// Now we can compute the "grading" of each vertex
	std::vector<std::pair<u32, float>> vertexGrade(this->get_vertex_count());
	for(std::size_t i = 0u; i < edges.size(); ++i) {
		auto currVertex = edges[i].first;
		float minCos = 1.f;

		// Find the end-index of edges
		std::size_t lastEdge;
		for(lastEdge = i; lastEdge < edges.size() && edges[lastEdge].first == currVertex; ++lastEdge) {}
		// Find the pairwise max. angle (== min. cosine) between the edges
		const auto v0 = points[currVertex];
		for(std::size_t e1 = i; (e1 + 1u) < lastEdge; ++e1) {
			const auto v1 = points[edges[e1].second];
			const auto v0v1 = ei::normalize(v1 - v0);
			for(std::size_t e2 = e1 + 1u; e2 < lastEdge; ++e2) {
				const auto v2 = points[edges[e2].second];
				const auto v0v2 = ei::normalize(v2 - v0);
				const auto cosTheta = ei::dot(v0v1, v0v2);
				minCos = std::min(minCos, cosTheta);
			}
		}

		// Compute and store the grade
		vertexGrade[currVertex] = std::make_pair(currVertex, std::cos(std::acos(minCos) / 2.f));
	}
	// Sort the vertices by grade (descending)
	std::sort(vertexGrade.begin(), vertexGrade.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

	// Clustering step: first create the grid
	// For this we discretize the bounding box in equal parts
	struct Cluster {
		ei::Vec3 centre;
		u32 count;
		OpenMesh::Geometry::Quadricf q;
	};
	struct Grid {
		std::optional<Cluster> cluster;
	};
	std::vector<u32> vertexGridIndices(this->get_vertex_count());
	std::vector<Grid> grids(ei::prod(gridRes));
	const auto aabbDiag = m_boundingBox.max - m_boundingBox.min;
	const auto get_grid_index = [gridRes](const ei::UVec3& gridPos) {
		return gridPos.x + gridPos.y * gridRes.x + gridPos.z * gridRes.y * gridRes.x;
	};
	for(const auto& vertex : vertexGrade) {
		const auto pos = points[vertex.first];
		// Check if the vertex is part of a cell already
		// For this, first find out the grid it belongs to and its neighbors
		std::optional<u32> chosenGridIndex = std::nullopt;
		float minDist = std::numeric_limits<float>::max();

		// Get the normalized position [0, 1]^3
		const auto normPos = (pos - m_boundingBox.min) / aabbDiag;
		// Get the discretized grid position with twice the grid resolution
		const auto gridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ 2u * gridRes } }, 2u * gridRes - 1u);
		const auto centreGridIdx = get_grid_index(gridPos / 2u);
		if(grids[centreGridIdx].cluster.has_value()) {
			chosenGridIndex = centreGridIdx;
			minDist = ei::len(pos - grids[centreGridIdx].cluster->centre);
		}

		// The other grid indices we get by adding/subtracting one if the double grid position is odd/even
		const auto compareGrid = [&grids, &minDist, &chosenGridIndex, centreGridIdx, pos](const i32 indexOffset) {
			const auto currGridIndex = centreGridIdx + indexOffset;
			// Check if the grid cell even has a cluster associated with it
			if(grids[currGridIndex].cluster.has_value()) {
				// We compare the distance between vertex and cluster instead of using cluster weight
				const auto distance = ei::len(pos - grids[currGridIndex].cluster->centre);
				if(distance < minDist) {
					minDist = distance;
					chosenGridIndex = currGridIndex;
				}
			}
		};

		// Iterate all neighbor cells
		constexpr ei::UVec3 offsets[] = {
			ei::UVec3{ 1u, 0u, 0u }, ei::UVec3{ 0u, 1u, 0u }, ei::UVec3{ 0u, 0u, 1u },
			ei::UVec3{ 1u, 1u, 0u }, ei::UVec3{ 1u, 0u, 1u }, ei::UVec3{ 0u, 1u, 1u },
			ei::UVec3{ 1u, 1u, 1u }
		};
		for(std::size_t i = 0u; i < sizeof(offsets) / sizeof(*offsets); ++i) {
			unsigned offset = 1u;
			for(std::size_t j = 0u; j < 3u; ++j) {
				if(offsets[j] != 0u) {
					if(gridPos[j] & 1u == 0u) {
						if(gridPos[j] > 0u)
							compareGrid(-static_cast<i32>(offset));
					} else if(gridPos[j] < gridRes[j] * 2u - 1u) {
						compareGrid(static_cast<i32>(offset));
					}
				}
				offset *= gridRes[j];
			}
		}

		// If vertex doesn't fit into any cell, create the cell with the vertex as centre
		if(!chosenGridIndex.has_value()) {
			grids[centreGridIdx].cluster = Cluster{ pos, 0u, {} };
			chosenGridIndex = centreGridIdx;
		}
		grids[chosenGridIndex.value()].cluster->count += 1u;
		grids[chosenGridIndex.value()].cluster->q += quadrics[vertex.first];
		vertexGridIndices[vertex.first] = chosenGridIndex.value();
	}

	// Compute the cluster centres with error quadrics
	for(auto& grid : grids) {
		if(grid.cluster.has_value()) {
			// Attempt to compute the optimal contraction point by inverting the quadric matrix
			const auto q = grid.cluster->q;
			const ei::Mat4x4 w{
				q.a(), q.b(), q.c(), q.d(),
				q.b(), q.e(), q.f(), q.g(),
				q.c(), q.f(), q.h(), q.i(),
				0,	   0,	  0,	 1
			};
			const auto inverse = invert_opt(w);
			if(inverse.has_value())
				grid.cluster->centre = ei::Vec3{ inverse.value() * ei::Vec4{ 0.f, 0.f, 0.f, 1.f } };
			else
				grid.cluster->centre /= static_cast<float>(grid.cluster->count);
		}
	}

	// Now we have to delete triangles which have degenerated into points/lines
	// Quads can also be transformed into triangles
	auto* indices = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices.get();
	auto* triIndices = indices;
	auto* quadIndices = indices + 3u * this->get_triangle_count();
	std::size_t triCount = this->get_triangle_count();
	std::size_t quadCount = this->get_quad_count();
	for(std::size_t i = 0u; i < this->get_triangle_count(); ++i) {
		u32* tri = triIndices + 3u * i;
		// Check if all three vertices are in different cells, otherwise mark the triangle as invalid
		if(vertexGridIndices[tri[0u]] != vertexGridIndices[tri[1u]]
		   && vertexGridIndices[tri[0u]] != vertexGridIndices[tri[2u]]) {
			tri[0u] = std::numeric_limits<u32>::max();
			triCount -= 1u;
		}
	}
	std::size_t quadTurnedTris = 0u;
	for(std::size_t i = 0u; i < this->get_quad_count(); ++i) {
		u32* quad = quadIndices + 4u * i;
		ei::UVec4 quadVertIndices{
			vertexGridIndices[quad[0u]], vertexGridIndices[quad[1u]],
			vertexGridIndices[quad[2u]], vertexGridIndices[quad[3u]]
		};
		// Check for tri
		// For triangles, we mark the outlying vertex, for edges/vertices we mark first and second
		if(quadVertIndices.x == quadVertIndices.y) {
			// At least edge
			if(quadVertIndices.x == quadVertIndices.z) {
				// At least triangle
				if(quadVertIndices.x != quadVertIndices.w) {
					// Mark as triangle
					quad[3u] = std::numeric_limits<u32>::max();
					quadCount -= 1u;
					quadTurnedTris += 1u;
				}
			} else {
				// Pre-mark as triangle
				quadCount -= 1u;
				// Check if it is actually triangle, mark as removal otherwise
				if(quadVertIndices.x != quadVertIndices.w) {
					quad[0u] = std::numeric_limits<u32>::max();
					quad[1u] = std::numeric_limits<u32>::max();
				} else {
					quad[2u] = std::numeric_limits<u32>::max();
					quadTurnedTris += 1u;
				}
			}
		} else {
			// No longer quad, but maybe triangle
			quadCount -= 1u;

			if(quadVertIndices.x == quadVertIndices.z) {
				quad[1u] = std::numeric_limits<u32>::max();
				if(quadVertIndices.x != quadVertIndices.w)
					quad[0u] = std::numeric_limits<u32>::max();
				else
					quadTurnedTris += 1u;
			} else if(quadVertIndices.y == quadVertIndices.z) {
				quad[0u] = std::numeric_limits<u32>::max();
				if(quadVertIndices.y != quadVertIndices.w)
					quad[1u] = std::numeric_limits<u32>::max();
				else
					quadTurnedTris += 1u;
			} else {
				quad[0u] = std::numeric_limits<u32>::max();
				quad[1u] = std::numeric_limits<u32>::max();
			}
		}
	}
	// Now we have to iterate the index buffer to swap/remove all remaining faces
	auto endTri = m_triangles;
	// Find the first valid triangle (from the back)
	while(endTri > 0u) {
		if()
	}
	// TODO: find last valid tri/quad
	for(std::size_t i = 0u; i < triCount; ++i) {
		if(triIndices[3u * i] == std::numeric_limits<u32>::max()) {
			// Swap last triangle
			endTri -= 1u;
			m_faceAttributes.copy(endTri, i);
			std::memcpy(triIndices + 3u * i, triIndices + 3u * endTri, 3u * sizeof(u32));
		}
	}

	auto endQuad = m_quads;
	// For quads we do two things - first we copy out all the quad->triangle ones, and then swap around
	const auto fillQuadHole = [&](const auto index) {
		if(quadIndices[4u * index] == std::numeric_limits<u32>::max()) {
			if(quadIndices[4u * index + 1u] == std::numeric_limits<u32>::max()) {
				// Swap last quad
				endQuad -= 1u;
				m_faceAttributes.copy(endQuad, m_triangles + index);
				std::memcpy(quadIndices + 4u * index, quadIndices + endQuad * 4u, 4u * sizeof(u32));
			} else {
				// Degenerated to triangle
				m_faceAttributes.copy(m_triangles + index, triCount);
				std::memcpy(triIndices + 3u * triCount, quadIndices + 4u * index + 1u, 3u * sizeof(u32));
				triCount += 1u;
			}
		} else {
			// Check the other vertices for triangle degeneration
			for(std::size_t j = 1u; j < 4u; ++j) {
				if(quadIndices[4u * index + j] == std::numeric_limits<u32>::max()) {
					m_faceAttributes.copy(m_triangles + index, triCount);
					// Copy over the indices (ignore the index marked invalid)
					auto* copyIndexTarget = triIndices + 3u * triCount;
					for(std::size_t c = 0u; c < 4u; ++c) {
						if(c != j)
							*(copyIndexTarget++) = quadIndices[4u * index + c];
					}
					triCount += 1u;
					break;
				}
			}
		}
	};

	for(std::size_t i = 0u; i < quadCount + quadTurnedTris; ++i) {
		if(quadIndices[4u * i] == std::numeric_limits<u32>::max()) {
			if(quadIndices[4u * i + 1u] == std::numeric_limits<u32>::max()) {
				// Swap last quad
				endQuad -= 1u;
				m_faceAttributes.copy(endQuad, m_triangles + i);
				std::memcpy(quadIndices + 4u * i, quadIndices + endQuad * 4u, 4u * sizeof(u32));
			} else {
				// Degenerated to triangle
				m_faceAttributes.copy(m_triangles + i, triCount);
				std::memcpy(triIndices + 3u * triCount, quadIndices + 4u * i + 1u, 3u * sizeof(u32));
				triCount += 1u;
			}
		} else {
			// Check the other vertices for triangle degeneration
			for(std::size_t j = 1u; j < 4u; ++j) {
				if(quadIndices[4u * i + j] == std::numeric_limits<u32>::max()) {
					m_faceAttributes.copy(m_triangles + i, triCount);
					// Copy over the indices (ignore the index marked invalid)
					auto* copyIndexTarget = triIndices + 3u * triCount;
					for(std::size_t c = 0u; c < 4u; ++c) {
						if(c != j)
							*(copyIndexTarget++) = quadIndices[4u * i + c];
					}
					triCount += 1u;
					break;
				}
			}
		}
	}

	// Resize the buffer
	auto& indexBuffer = m_indexBuffer.template get<IndexBuffer<Device::CPU>>();
	indexBuffer.indices.reset(Allocator<Device::CPU>::realloc(indexBuffer.indices.release(), indexBuffer.reserved,
															  3u * triCount + 4u * quadCount));

	this->mark_changed(Device::CPU);
}
#endif // 0

void Polygons::after_tessellation(const PolygonMeshType& mesh, const OpenMesh::FaceHandle tempHandle,
								  const OpenMesh::FPropHandleT<OpenMesh::FaceHandle>& oldFaceProp) {
	// We have to update the attributes. Two parts:
	// - For vertices, only new ones, none get deleted, standard props (UV, normal etc.) are already computed in mesh:
	//   we only update those, other attributes are left as default (have to be computed manually if desired).
	// - Faces not only get added, but the replaced faces get deleted. For those we copy over all attributes.
	//   After that we have to compact the attributes and recompute the index buffer.
	// Don't forget to remove the temporary face vertices from the total count
	m_vertexAttributes.resize(mesh.n_vertices() - 3u);

	// First we have to allocate extra faces to have swap space
	const auto faceCount = mesh.n_faces() - 1u;
	m_faceAttributes.resize(faceCount);
	for(std::size_t i = this->get_face_count(); i < mesh.n_faces(); ++i) {
		if(i == static_cast<std::size_t>(tempHandle.idx()))
			continue;
		// We don't need to check previously existing faces
		const auto face = mesh.face_handle(static_cast<unsigned>(i));
		if(const auto oldFace = mesh.property(oldFaceProp, face); oldFace.is_valid()) {
			// Make sure we ignore the temporary face
			const auto newFaceIdx = (i < static_cast<std::size_t>(tempHandle.idx())) ? i : (i - 1u);
			m_faceAttributes.copy(static_cast<std::size_t>(oldFace.idx()),
											   newFaceIdx);
		}
	}
	const auto* matIndices = this->template acquire_const<Device::CPU, MaterialIndex>(m_matIndicesHdl);
	// Remove old faces. There must be more new faces than removed ones by design
	for(std::size_t i = 0u; i < this->get_face_count(); ++i) {
		const auto face = mesh.face_handle(static_cast<unsigned>(i));
		if(mesh.status(face).deleted()) {
			const auto currSize = m_faceAttributes.get_attribute_elem_count();
			m_faceAttributes.copy(currSize - 1u, i);
			m_faceAttributes.resize(currSize - 1u);
		}
	}
	// Copy over the default vertex attributes; don't forget to ignore the temporary vertices
	auto* points = this->template acquire<Device::CPU, ei::Vec3>(this->get_points_hdl());
	auto* normals = this->template acquire<Device::CPU, ei::Vec3>(this->get_normals_hdl());
	auto* uvs = this->template acquire<Device::CPU, ei::Vec2>(this->get_uvs_hdl());
	std::size_t vertexIndex = 0u;
	for(const auto vertex : mesh.vertices()) {
		points[vertexIndex] = util::pun<ei::Vec3>(mesh.point(vertex));
		normals[vertexIndex] = util::pun<ei::Vec3>(mesh.normal(vertex));
		uvs[vertexIndex] = util::pun<ei::Vec2>(mesh.texcoord2D(vertex));
		vertexIndex += 1u;
	}

	m_faceAttributes.shrink_to_fit();
	// Recompute the index buffer
	this->rebuild_index_buffer(mesh, tempHandle);
}

void Polygons::rebuild_index_buffer(const PolygonMeshType& mesh, const OpenMesh::FaceHandle tempHandle) {
	// Recompute the index buffer. For this we have to ignore deleted faces
	this->mark_changed(Device::CPU);
	// Count 
	m_triangles = 0u;
	m_quads = 0u;
	for(const auto face : mesh.faces()) {
		const auto vBegin = mesh.cfv_ccwbegin(face);
		const auto vEnd = mesh.cfv_ccwend(face);
		const auto vertexCount = std::distance(vBegin, vEnd);
		if(vertexCount == 0)
			continue;
		if(vertexCount == 3u)
			++m_triangles;
		else if(vertexCount == 4u)
			++m_quads;
		else
			throw std::runtime_error("Tessellation added a non-quad/tri face (" + std::to_string(vertexCount) + " vertices)");
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

	// Fetch the vertex handles of the temporary face, since those have to be deleted as well
	auto vTempIter = mesh.cfv_ccwbegin(tempHandle);
	const auto tempV0 = *vTempIter;
	const auto tempV1 = *(++vTempIter);
	const auto tempV2 = *(++vTempIter);

	// Rebuild the index buffer
	this->reserve_index_buffer<Device::CPU>(3u * m_triangles + 4u * m_quads);
	std::size_t currTri = 0u;
	std::size_t currQuad = 0u;
	auto* indexBuffer = m_indexBuffer.template get<IndexBuffer<Device::CPU>>().indices.get();

	// Keep track of the last after-delete face inserted
	// We do this because the prior step of rebuilding index buffer was to fill in the
	// holes of deleted faces in the attributes
	auto currEnd = mesh.n_faces() - 1u;
	for(auto iter = mesh.faces_begin(); iter != mesh.faces_end(); ++iter) {
		auto face = iter.handle();
		if(face == tempHandle)
			continue;
		if(static_cast<std::size_t>(face.idx()) > currEnd)
			break;
		if(mesh.status(face).deleted()) {
			face = mesh.face_handle(static_cast<unsigned>(currEnd));
			currEnd -= 1u;
		}

		const auto startVertex = mesh.cfv_ccwbegin(face);
		const auto endVertex = mesh.cfv_ccwend(face);
		const auto vertexCount = std::distance(startVertex, endVertex);

		u32* currIndices = indexBuffer;
		if(vertexCount == 3u) {
			currIndices += 3u * currTri++;
		} else {
			currIndices += 3u * m_triangles + 4u * currQuad++;
		}
		for(auto vertexIter = startVertex; vertexIter != endVertex; ++vertexIter) {
			const auto vertexIdx = static_cast<u32>(vertexIter->idx());
			auto index = vertexIdx;
			if(vertexIdx > static_cast<u32>(tempV0.idx()))
				index -= 1u;
			if(vertexIdx > static_cast<u32>(tempV1.idx()))
				index -= 1u;
			if(vertexIdx > static_cast<u32>(tempV2.idx()))
				index -= 1u;
			mAssert(index < m_vertexAttributes.get_attribute_elem_count());
			*(currIndices++) = index;
			const auto pt = util::pun<ei::Vec3>(mesh.point(*vertexIter));
			m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ pt } };
		}
	}
}

void Polygons::tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
						  const bool usePhong) {
	auto profileTimer = Profiler::core().start<CpuProfileState>("Polygons::tessellate");
	auto mesh = this->create_halfedge_structure();
	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;

	// This is necessary since we'd otherwise need to pass an accessor into the tessellater
	tessellation::Tessellater tessellater(oracle);
	tessellater.set_phong_tessellation(usePhong);

	if(scenario != nullptr) {
		// Add the material indices as mesh property
		OpenMesh::FPropHandleT<MaterialIndex> matIdxProp;
		mesh.add_property(matIdxProp);
		const auto* matIndices = this->template acquire_const<Device::CPU, MaterialIndex>(this->get_material_indices_hdl());
		copy(mesh.property(matIdxProp).data_vector().data(), matIndices, sizeof(MaterialIndex) * this->get_face_count());
		oracle.set_mat_properties(*scenario, matIdxProp);
	}

	mesh.request_face_status();
	mesh.request_vertex_status();
	const auto tempFace = tessellater.tessellate(mesh);
	const auto oldFaceProp = tessellater.get_old_face_property();
	this->after_tessellation(mesh, tempFace, oldFaceProp);
	m_wasDisplaced = true;
	logInfo("Tessellated polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

void Polygons::displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario) {
	auto profileTimer = Profiler::core().start<CpuProfileState>("Polygons::displace");
	auto mesh = this->create_halfedge_structure();
	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;

	// This is necessary since we'd otherwise need to pass an accessor into the tessellater
	// Add the material indices as mesh property
	OpenMesh::FPropHandleT<MaterialIndex> matIdxProp;
	mesh.add_property(matIdxProp);
	const auto* matIndices = this->template acquire_const<Device::CPU, MaterialIndex>(this->get_material_indices_hdl());
	copy(mesh.property(matIdxProp).data_vector().data(), matIndices, sizeof(MaterialIndex) * this->get_face_count());
	oracle.set_mat_properties(scenario, matIdxProp);
	tessellation::DisplacementMapper tessellater(oracle);
	tessellater.set_scenario(scenario);
	tessellater.set_material_idx_hdl(matIdxProp);
	tessellater.set_phong_tessellation(true);

	mesh.request_face_status();
	mesh.request_vertex_status();
	const auto tempFace = tessellater.tessellate(mesh);
	const auto oldFaceProp = tessellater.get_old_face_property();
	this->after_tessellation(mesh, tempFace, oldFaceProp);
	m_wasDisplaced = true;
	logInfo("Displaced polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

bool Polygons::apply_animation(u32 frame, const Bone* bones) {
	if(!m_animationWeightHdl.has_value())
		return false;
	auto* weights = acquire_const<Device::CPU, ei::UVec4>(*m_animationWeightHdl);
	auto* normals = acquire<Device::CPU, ei::Vec3>(get_normals_hdl());
	auto* positions = acquire<Device::CPU, ei::Vec3>(get_points_hdl());

	// We also have to recalculate the bounding box
	m_boundingBox.min = ei::Vec3{ std::numeric_limits<float>::max() };
	m_boundingBox.max = ei::Vec3{ -std::numeric_limits<float>::max() };

	for(std::size_t vertex = 0u; vertex < this->get_vertex_count(); ++vertex) {
		// Decode weight and bone index and sum all the dual quaternions
		ei::UVec4 codedWeights = weights[vertex];
		// We must NOT start at 0,0,0,1 for the real part
		ei::DualQuaternion q{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		bool transformed = false;
		for(int i = 0; i < 4; ++i) {
			float weight = (codedWeights[i] >> 22) / 1023.0f;
			u32 idx = codedWeights[i] & 0x003fffff;
			if(idx != 0x003fffff) {
				q += bones[idx].transformation * weight;
				transformed = true;
			}
		}
		if(transformed)
			q = ei::normalize(q);
		else
			q = ei::qqidentity();
		const auto newPos = ei::transform(positions[vertex], q);
		positions[vertex] = newPos;
		normals[vertex] = ei::transformDir(normals[vertex], q);
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ newPos } };
	}
	this->mark_changed(Device::CPU);
	return true;
}

void Polygons::transform(const ei::Mat3x4& transMat) {
	this->synchronize<Device::CPU>();
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
	ei::Vec3* vertices = m_vertexAttributes.template acquire<Device::CPU, ei::Vec3>(m_pointsHdl);
	for(size_t i = 0; i < this->get_vertex_count(); i++) {
		vertices[i] = rotation * vertices[i];
		vertices[i] += translation;
		m_boundingBox.max = ei::max(vertices[i], m_boundingBox.max);
		m_boundingBox.min = ei::min(vertices[i], m_boundingBox.min);
	}
	// Transform normals
	ei::Vec3* normals = m_vertexAttributes.template acquire<Device::CPU, ei::Vec3>(m_normalsHdl);
	for(size_t i = 0; i < this->get_vertex_count(); i++) { // one normal per vertex
		normals[i] = ei::normalize(rotation * normals[i]);
	}
	m_vertexAttributes.mark_changed(Device::CPU);
}

float Polygons::compute_surface_area() {
	this->template synchronize<Device::CPU>();
	float area = 0.f;
	const auto* indexBuffer = this->template get_index_buffer<Device::CPU>();
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	// Faces are split into triangles and quads
	for(std::size_t tri = 0u; tri < m_triangles; ++tri) {
		const ei::UVec3 indices{
			indexBuffer[3u * tri + 0u],
			indexBuffer[3u * tri + 1u],
			indexBuffer[3u * tri + 2u]
		};
		const auto pA = points[indices.x];
		const auto pB = points[indices.y];
		const auto pC = points[indices.z];
		area += ei::len(ei::cross(pB - pA, pC - pA));
	}
	for(std::size_t quad = 0u; quad < m_quads; ++quad) {
		const ei::UVec4 indices{
			indexBuffer[3u * m_triangles + 4u * quad + 0u],
			indexBuffer[3u * m_triangles + 4u * quad + 1u],
			indexBuffer[3u * m_triangles + 4u * quad + 2u],
			indexBuffer[3u * m_triangles + 4u * quad + 3u]
		};
		const auto pA = points[indices.x];
		const auto pB = points[indices.y];
		const auto pC = points[indices.z];
		const auto pD = points[indices.w];
		// Assume that the quad is planar
		area += ei::len(ei::cross(pB - pA, pC - pA));
		area += ei::len(ei::cross(pC - pA, pD - pA));
	}
	return 0.5f * area;
}

void Polygons::remove_curvature() {
	u32 oldValue = m_curvRefCount.load(std::memory_order_acquire);
	u32 newValue;
	do {
		newValue = std::max(1u, oldValue) - 1u;
	} while(m_curvRefCount.compare_exchange_weak(oldValue, newValue, std::memory_order_acq_rel));
	if(newValue == 0u && m_curvatureHdl.has_value()) {
		m_vertexAttributes.remove(m_curvatureHdl.value());
		m_curvatureHdl.reset();
	}
}

void Polygons::compute_curvature() {
#if 0
	// Check if the curvature has been computed before
	if(m_curvRefCount.fetch_add(1u) == 0 && !m_curvatureHdl.has_value())
		m_curvatureHdl = this->template add_vertex_attribute<float>("mean_curvature");

	// TODO: we need neighborhood information

	auto* curv = m_vertexAttributes.template acquire<Device::CPU, float>(m_curvatureHdl.value());
	const auto* points = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_points_hdl());
	const auto* normals = this->template acquire_const<Device::CPU, ei::Vec3>(this->get_normals_hdl());

	for(std::size_t vertex = 0u; vertex < this->get_vertex_count(); ++vertex) {
		// Fetch data at reference vertex and create an orthogonal space
		const auto vPos = points[vertex];
		const auto vNrm = normals[vertex];
		const auto dirU = normalize(perpendicular(vNrm));
		const auto dirV = cross(vNrm, dirU);
		// Construct an equation system which fits a paraboloid to the 1-ring.
		// We need Aᵀ A x = Aᵀ b for the least squares system.
		// Interestingly, it suffices to solve for two of the variables instead of
		// all three, if we are interested in mean curvature only.
		ei::Mat2x2 ATA { 0.0f };
		ei::Vec2 ATb { 0.0f };
		// For each edge add an equation
		for(auto vi = m_meshData->vv_ccwiter(v); vi.is_valid(); ++vi) {
			Point viPos = ei::details::hard_cast<ei::Vec3>(m_meshData->point(*vi));
			Direction viNrm = ei::details::hard_cast<ei::Vec3>(m_meshData->normal(*vi));
			ei::Vec3 edge = vPos - viPos;
			// Normal curvature fit with different curvature estimations
			const ei::Vec2 y { dot(dirU, edge), dot(dirV, edge) };
			const float normSq = y.x * y.x + y.y * y.y + 1e-30f;
			//const float ki = 2.0f * dot(vNrm, edge) / (dot(edge, edge) + 1e-30f);		// Circular
			//const float ki = dot(vNrm - viNrm, edge) / (dot(edge, edge) + 1e-30f);	// Circular projected
			//const float ki = 2.0f * dot(vNrm, edge) / normSq;							// Parabolic
			const float ki = dot(vNrm - viNrm, edge) / normSq;						// Parabolic projected
			float a = y.x * y.x / normSq;
			float c = y.y * y.y / normSq;

			ATb += ki * ei::Vec2{ a, c };
			ATA += ei::Mat2x2{ a*a, a*c, a*c, c*c };
		}
		// Solve with least squares
		float detA = determinant(ATA);
		float e = ATb.x * ATA.m11 - ATb.y * ATA.m01;
		float g = ATA.m00 * ATb.y - ATA.m10 * ATb.x;
		detA += (ATA.m00 + ATA.m11) * 1e-7f + 1e-30f; // Regularize with the trace
		float meanc = (e + g) * 0.5f / detA;
		mAssert(!std::isnan(meanc));
		curv[v.idx()] = meanc;
	}
#endif // 0
}


template < Device dev >
void Polygons::synchronize() {
	m_vertexAttributes.synchronize<dev>();
	m_faceAttributes.synchronize<dev>();
	// Synchronize the index buffer

	if(m_indexBuffer.template get<IndexBuffer<dev>>().indices.get() == nullptr) {
		// Try to find a valid device
		bool synced = false;
		m_indexBuffer.for_each([&](auto& buffer) {
			using ChangedBuffer = std::decay_t<decltype(buffer)>;
			if(!synced && buffer.indices.get() != nullptr) {
				this->synchronize_index_buffer<ChangedBuffer::DEVICE, dev>();
				synced = true;
			}
		});
	}
}

void Polygons::mark_changed(Device dev) {
	get_attributes<true>().mark_changed(dev);
	get_attributes<false>().mark_changed(dev);
	if(dev != Device::CPU)
		unload_index_buffer<Device::CPU>();
	if(dev != Device::CUDA)
		unload_index_buffer<Device::CUDA>();
	if(dev != Device::OPENGL)
		unload_index_buffer<Device::OPENGL>();
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
		this->template acquire_const<dev, ei::Vec3>(this->get_points_hdl()),
		this->template acquire_const<dev, ei::Vec3>(this->get_normals_hdl()),
		this->template acquire_const<dev, ei::Vec2>(this->get_uvs_hdl()),
		this->template acquire_const<dev, u16>(this->get_material_indices_hdl()),
		this->template get_index_buffer<dev>(),
		//	m_attribBuffer.template get<AttribBuffer<dev>>().vertex,
		ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{},
		ConstArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>>{}
	};
}

template < Device dev >
void Polygons::update_attribute_descriptor(PolygonsDescriptor<dev>& descriptor,
										   const std::vector<AttributeIdentifier>& vertexAttribs,
										   const std::vector<AttributeIdentifier>& faceAttribs) {
	this->synchronize<dev>();

	// Free the previous attribute array if no attributes are wanted
	auto& buffer = m_attribBuffer.template get<AttribBuffer<dev>>();

	if(vertexAttribs.size() == 0)
		buffer.vertex.reset();
	else // else keep current nullptr
		buffer.vertex = make_udevptr_array<dev, ArrayDevHandle_t<dev, void>, false>(vertexAttribs.size());
	
	if(faceAttribs.size() == 0)
		buffer.face.reset();
	else // else keep current nullptr
		buffer.face = make_udevptr_array<dev, ArrayDevHandle_t<dev, void>, false>(faceAttribs.size());

	std::vector<ArrayDevHandle_t<dev, void>> cpuVertexAttribs;
	std::vector<ArrayDevHandle_t<dev, void>> cpuFaceAttribs;
	cpuVertexAttribs.reserve(vertexAttribs.size());
	cpuFaceAttribs.reserve(faceAttribs.size());
	for(const auto& ident : vertexAttribs)
		cpuVertexAttribs.push_back(this->template acquire_vertex<dev, void>(ident));
	for(const auto& ident : faceAttribs)
		cpuFaceAttribs.push_back(this->template acquire_face<dev, void>(ident));
	copy(buffer.vertex.get(), cpuVertexAttribs.data(), sizeof(cpuVertexAttribs.front()) * vertexAttribs.size());
	copy(buffer.face.get(), cpuFaceAttribs.data(), sizeof(cpuFaceAttribs.front()) * faceAttribs.size());

	descriptor.numVertexAttributes = static_cast<u32>(vertexAttribs.size());
	descriptor.numFaceAttributes = static_cast<u32>(faceAttribs.size());
	descriptor.vertexAttributes = buffer.vertex.get();
	descriptor.faceAttributes = buffer.face.get();
}

// Reserves more space for the index buffer
template < Device dev, bool markChanged >
void Polygons::reserve_index_buffer(std::size_t capacity) {
	auto& buffer = m_indexBuffer.get<IndexBuffer<dev>>();
	if(capacity > buffer.reserved) {
		buffer.indices = make_udevptr_array<dev, u32, false>(capacity);
		buffer.reserved = capacity;
		if constexpr(markChanged) {
			m_indexBuffer.for_each([](auto& data) {
				using ChangedBuffer = std::decay_t<decltype(data)>;
				if(ChangedBuffer::DEVICE != dev && data.indices != nullptr) {
					data.indices.reset();
					data.reserved = 0u;
				}
			});
		}
	}
}

// Synchronizes two device index buffers
template < Device changed, Device sync >
void Polygons::synchronize_index_buffer() {
	if constexpr(changed != sync) {
		auto& changedBuffer = m_indexBuffer.get<IndexBuffer<changed>>();
		auto& syncBuffer = m_indexBuffer.get<IndexBuffer<sync>>();

		// Check if we need to realloc
		if(syncBuffer.reserved < m_triangles + m_quads)
			this->reserve_index_buffer<sync, false>(3u * m_triangles + 4u * m_quads);

		if(changedBuffer.reserved != 0u)
			copy(syncBuffer.indices.get(), changedBuffer.indices.get(), sizeof(u32) * (3u * m_triangles + 4u * m_quads));
	}
}

template < Device dev >
void Polygons::unload_index_buffer() {
	auto& idxBuffer = m_indexBuffer.template get<IndexBuffer<dev>>();
	idxBuffer.indices.reset();
	idxBuffer.reserved = 0u;
}

template < Device dev >
void Polygons::resizeAttribBuffer(std::size_t v, std::size_t f) {
	AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
	// Resize the attribute array if necessary
	if(attribBuffer.faceSize < f) {
		attribBuffer.face = make_udevptr_array<dev, ArrayDevHandle_t<dev, void>, false>(f);
		attribBuffer.faceSize = f;
	}
	if(attribBuffer.vertSize < v) {
		attribBuffer.vertex = make_udevptr_array<dev, ArrayDevHandle_t<dev, void>, false>(v);
		attribBuffer.vertSize = v;
	}
}

// Explicit instantiations
template void Polygons::reserve_index_buffer<Device::CPU, true>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::CUDA, true>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::OPENGL, true>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::CPU, false>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::CUDA, false>(std::size_t capacity);
template void Polygons::reserve_index_buffer<Device::OPENGL, false>(std::size_t capacity);
template void Polygons::synchronize_index_buffer<Device::CPU, Device::CUDA>();
template void Polygons::synchronize_index_buffer<Device::CPU, Device::OPENGL>();
template void Polygons::synchronize_index_buffer<Device::CUDA, Device::CPU>();
//template void Polygons::synchronize_index_buffer<Device::CUDA, Device::OPENGL>();
//template void Polygons::synchronize_index_buffer<Device::OPENGL, Device::CPU>();
//template void Polygons::synchronize_index_buffer<Device::OPENGL, Device::CUDA>();
template void Polygons::unload_index_buffer<Device::CPU>();
template void Polygons::unload_index_buffer<Device::CUDA>();
template void Polygons::unload_index_buffer<Device::OPENGL>();
template void Polygons::resizeAttribBuffer<Device::CPU>(std::size_t v, std::size_t f);
template void Polygons::resizeAttribBuffer<Device::CUDA>(std::size_t v, std::size_t f);
template void Polygons::resizeAttribBuffer<Device::OPENGL>(std::size_t v, std::size_t f);
template void Polygons::synchronize<Device::CPU>();
template void Polygons::synchronize<Device::CUDA>();
template void Polygons::synchronize<Device::OPENGL>();
template PolygonsDescriptor<Device::CPU> Polygons::get_descriptor<Device::CPU>();
template PolygonsDescriptor<Device::CUDA> Polygons::get_descriptor<Device::CUDA>();
template PolygonsDescriptor<Device::OPENGL> Polygons::get_descriptor<Device::OPENGL>();
template void Polygons::update_attribute_descriptor<Device::CPU>(PolygonsDescriptor<Device::CPU>& descriptor,
																 const std::vector<AttributeIdentifier>& vertexAttribs,
																 const std::vector<AttributeIdentifier>& faceAttribs);
template void Polygons::update_attribute_descriptor<Device::CUDA>(PolygonsDescriptor<Device::CUDA>& descriptor,
																  const std::vector<AttributeIdentifier>& vertexAttribs,
																  const std::vector<AttributeIdentifier>& faceAttribs);
template void Polygons::update_attribute_descriptor<Device::OPENGL>(PolygonsDescriptor<Device::OPENGL>& descriptor,
																	const std::vector<AttributeIdentifier>& vertexAttribs,
																	const std::vector<AttributeIdentifier>& faceAttribs);


} // namespace mufflon::scene::geometry
