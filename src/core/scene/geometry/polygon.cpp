#include "polygon.hpp"
#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/punning.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/data_structs/count_octree.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"
#include "core/math/curvature.hpp"
#include "core/scene/tessellation/tessellater.hpp"
#include "core/scene/tessellation/displacement_mapper.hpp"
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>

namespace mufflon::scene::geometry {

// Cluster for vertex clustering
struct VertexCluster {
	ei::Vec3 posAccum{ 0.f };
	u32 count{ 0u };
	OpenMesh::Geometry::Quadricf q{};

	void add_vertex(const ei::Vec3& pos, const OpenMesh::Geometry::Quadricf& quadric) noexcept {
		count += 1u;
		posAccum += pos;
		q += quadric;
	}
};

namespace {

// Invert function that returns optional to indicate non-invertability instead of identity matrix
std::optional<ei::Mat4x4> invert(const ei::Mat4x4& mat0) noexcept {
	ei::Mat4x4 LU;
	ei::UVec4 p;
	if(ei::decomposeLUp(mat0, LU, p))
		return ei::solveLUp(LU, p, ei::identity4x4());
	return std::nullopt;
}

template < class Iter >
u32 compute_cluster_center(const Iter begin, const Iter end) {
	u32 clusterCount = 0u;
	for(auto iter = begin; iter != end; ++iter) {
		auto& cluster = *iter;
		if(cluster.count > 0) {
			// Attempt to compute the optimal contraction point by inverting the quadric matrix
			const auto q = cluster.q;
			const ei::Mat4x4 w{
				q.a(), q.b(), q.c(), q.d(),
				q.b(), q.e(), q.f(), q.g(),
				q.c(), q.f(), q.h(), q.i(),
				0,	   0,	  0,	 1
			};
			const auto inverse = invert(w);
			if(inverse.has_value())
				cluster.posAccum = ei::Vec3{ inverse.value() * ei::Vec4{ 0.f, 0.f, 0.f, 1.f } };
			else
				cluster.posAccum /= static_cast<float>(cluster.count);
			clusterCount += 1;
		}
	}
	return clusterCount;
}

template < class Iter >
void compute_vertex_normals(PolygonMeshType& mesh, const Iter begin, const Iter end) {
	for(auto iter = begin; iter != end; ++iter) {
		const auto vertex = *iter;
		PolygonMeshType::Normal normal;
		// Ignoring warning about OpenMesh initializing a float vector with '0.0'...
#pragma warning(push)
#pragma warning(disable : 4244)
		mesh.calc_vertex_normal_correct(vertex, normal);
#pragma warning(pop)
		mesh.set_normal(vertex, normal);
	}
}

}

// Default construction, creates material-index attribute.
Polygons::Polygons() :
	m_meshData(std::make_unique<PolygonMeshType>()),
	m_vertexAttributes(*m_meshData),
	m_faceAttributes(*m_meshData),
	m_pointsHdl(m_vertexAttributes.register_point_attribute()),
	m_normalsHdl(m_vertexAttributes.register_normal_attribute()),
	m_uvsHdl(m_vertexAttributes.register_uv_attribute()),
	m_curvatureHdl{},
	m_matIndicesHdl{ this->template add_face_attribute<u16>("materials") }
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

Polygons::Polygons(const Polygons& poly) :
	m_meshData(std::make_unique<PolygonMeshType>(*poly.m_meshData)),
	m_vertexAttributes(*m_meshData),
	m_faceAttributes(*m_meshData),
	m_pointsHdl(m_vertexAttributes.register_point_attribute()),
	m_normalsHdl(m_vertexAttributes.register_normal_attribute()),
	m_uvsHdl(m_vertexAttributes.register_uv_attribute()),
	m_curvatureHdl{},
	m_matIndicesHdl{ poly.m_matIndicesHdl },
	m_boundingBox(poly.m_boundingBox),
	m_triangles(poly.m_triangles),
	m_quads(poly.m_quads)
{
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
	m_curvatureHdl(std::move(poly.m_curvatureHdl)),
	m_matIndicesHdl(std::move(poly.m_matIndicesHdl)),
	m_boundingBox(std::move(poly.m_boundingBox)),
	m_triangles(poly.m_triangles),
	m_quads(poly.m_quads)
{
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
			Allocator<ChangedBuffer::DEVICE>::template free<u32>(buffer.indices, buffer.reserved);
	});
	m_attribBuffer.for_each([&](auto& buffer) {
		using ChangedBuffer = std::decay_t<decltype(buffer)>;
		if(buffer.vertSize != 0)
			Allocator<ChangedBuffer::DEVICE>::template free<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.vertex, buffer.vertSize);
		if(buffer.faceSize != 0)
			Allocator<ChangedBuffer::DEVICE>::template free<ArrayDevHandle_t<ChangedBuffer::DEVICE, void>>(buffer.face, buffer.faceSize);
	});
}

std::size_t Polygons::add_bulk(const VertexAttributeHandle& hdl, const VertexHandle& startVertex,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
	return m_vertexAttributes.restore(hdl, attrStream, static_cast<std::size_t>(startVertex.idx()), count);
}

std::size_t Polygons::add_bulk(const FaceAttributeHandle& hdl, const FaceHandle& startFace,
							   std::size_t count, util::IByteReader& attrStream) {
	mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData->n_faces());
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

	m_vertexAttributes.template acquire<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl)[vh.idx()] = util::pun<OpenMesh::Vec3f>(point);
	m_vertexAttributes.template acquire<Device::CPU, OpenMesh::Vec3f>(m_normalsHdl)[vh.idx()] = util::pun<OpenMesh::Vec3f>(normal);
	m_vertexAttributes.template acquire<Device::CPU, OpenMesh::Vec2f>(m_uvsHdl)[vh.idx()] = util::pun<OpenMesh::Vec2f>(uv);
	m_vertexAttributes.mark_changed(Device::CPU);
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

	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_triangles;
	return hdl;
}

Polygons::TriangleHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
									   const VertexHandle& v2, MaterialIndex idx) {
	TriangleHandle hdl = this->add(v0, v1, v2);
	m_faceAttributes.template acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl.idx()] = idx;
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
	// TODO: slow, hence replace with reserve
	m_faceAttributes.resize(m_faceAttributes.get_attribute_elem_count() + 1u);
	++m_quads;
	return hdl;
}

Polygons::QuadHandle Polygons::add(const VertexHandle& v0, const VertexHandle& v1,
								   const VertexHandle& v2, const VertexHandle& v3,
								   MaterialIndex idx) {
	QuadHandle hdl = this->add(v0, v1, v2, v3);
	m_faceAttributes.template acquire<Device::CPU, MaterialIndex>(m_matIndicesHdl)[hdl.idx()] = idx;
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
	const OpenMesh::Vec3f* points = m_vertexAttributes.template acquire_const<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl);
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
	const OpenMesh::Vec3f* points = m_vertexAttributes.template acquire_const<Device::CPU, OpenMesh::Vec3f>(m_pointsHdl);
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

void Polygons::tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
						  const bool usePhong) {
	auto profileTimer = Profiler::core().start<CpuProfileState>("Polygons::tessellate");
	this->synchronize<Device::CPU>();
	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;

	// This is necessary since we'd otherwise need to pass an accessor into the tessellater
	tessellation::Tessellater tessellater(oracle);
	tessellater.set_phong_tessellation(usePhong);

	if(scenario != nullptr) {
		OpenMesh::FPropHandleT<MaterialIndex> matIdxProp;
		m_meshData->get_property_handle(matIdxProp, MAT_INDICES_NAME);
		oracle.set_mat_properties(*scenario, matIdxProp);
	}

	tessellater.tessellate(*m_meshData);
	this->mark_changed(Device::CPU);
	this->rebuild_index_buffer();
	m_wasDisplaced = true;
	logInfo("Tessellated polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

void Polygons::displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario) {
	auto profileTimer = Profiler::core().start<CpuProfileState>("Polygons::displace");
	this->synchronize<Device::CPU>();
	// Then perform tessellation
	const std::size_t prevTri = m_triangles;
	const std::size_t prevQuad = m_quads;
	// This is necessary since we'd otherwise need to pass an accessor into the tessellater
	OpenMesh::FPropHandleT<MaterialIndex> matIdxProp{ static_cast<int>(m_matIndicesHdl.index) };
	oracle.set_mat_properties(scenario, matIdxProp);
	tessellation::DisplacementMapper tessellater(oracle);
	tessellater.set_scenario(scenario);
	tessellater.set_material_idx_hdl(matIdxProp);

	tessellater.set_phong_tessellation(true);
	tessellater.tessellate(*m_meshData);

	this->mark_changed(Device::CPU);
	this->rebuild_index_buffer();
	m_wasDisplaced = true;
	logInfo("Uniformly tessellated polygon mesh (", prevTri, "/", prevQuad,
			" -> ", m_triangles, "/", m_quads, ")");
}

bool Polygons::apply_animation(u32 frame, const Bone* bones) {
	if(!m_animationWeightHdl.has_value())
		return false;
	auto weights = acquire_const<Device::CPU, ei::UVec4>(*m_animationWeightHdl);
	auto normals = acquire<Device::CPU, ei::Vec3>(get_normals_hdl());
	auto positions = acquire<Device::CPU, ei::Vec3>(get_points_hdl());

	// We also have to recalculate the bounding box
	m_boundingBox.min = ei::Vec3{ std::numeric_limits<float>::max() };
	m_boundingBox.max = ei::Vec3{ -std::numeric_limits<float>::max() };

	for(auto vertex : m_meshData->vertices()) {
		// Decode weight and bone index and sum all the dual quaternions
		ei::UVec4 codedWeights = weights[vertex.idx()];
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
		const auto newPos = ei::transform(positions[vertex.idx()], q);
		positions[vertex.idx()] = newPos;
		normals[vertex.idx()] = ei::transformDir(normals[vertex.idx()], q);
		m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ newPos } };
	}
	this->mark_changed(Device::CPU);
	return true;
}

template < class T >
void Polygons::compute_error_quadrics(OpenMesh::VPropHandleT<OpenMesh::Geometry::QuadricT<T>> quadricProps) {
	using OpenMesh::Geometry::QuadricT;
	for(const auto vertex : m_meshData->vertices())
		m_meshData->property(quadricProps, vertex).clear();
	for(const auto face : m_meshData->faces()) {
		// Assume triangle
		auto vIter = m_meshData->cfv_ccwbegin(face);
		const auto vh0 = *vIter;
		const auto vh1 = *(++vIter);
		const auto vh2 = *(++vIter);

		const auto p0 = util::pun<ei::Vec3>(m_meshData->point(vh0));
		const auto p1 = util::pun<ei::Vec3>(m_meshData->point(vh1));
		const auto p2 = util::pun<ei::Vec3>(m_meshData->point(vh2));
		auto normal = ei::cross(p1 - p0, p2 - p0);
		auto area = ei::len(normal);
		if(area > std::numeric_limits<decltype(area)>::min()) {
			normal /= area;
			area *= 0.5f;
		}

		const auto d = -ei::dot(p0, normal);
		QuadricT<T> q{ normal.x, normal.y, normal.z, d };
		q *= area;
		m_meshData->property(quadricProps, vh0) += q;
		m_meshData->property(quadricProps, vh1) += q;
		m_meshData->property(quadricProps, vh2) += q;
	}
}

// Creates a decimater 
OpenMesh::Decimater::DecimaterT<PolygonMeshType> Polygons::create_decimater() {
	return OpenMesh::Decimater::DecimaterT<PolygonMeshType>(*m_meshData);
}

std::size_t Polygons::cluster(const data_structs::CountOctree& octree,
							  const std::size_t targetVertices,
							  const bool garbageCollect) {
	using OpenMesh::Geometry::Quadricf;

	this->synchronize<Device::CPU>();

	const auto previousVertices = m_meshData->n_vertices();

	// Perform a breadth-first search to bring clusters down in equal levels
	std::vector<bool> octreeNodeMask(octree.capacity(), false);
	std::vector<data_structs::CountOctree::NodeId> currLevel;
	std::vector<data_structs::CountOctree::NodeId> nextLevel;
	currLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(targetVertices, 1llu))));
	nextLevel.reserve(static_cast<std::size_t>(ei::log2(std::max(targetVertices, 1llu))));
	std::size_t finalNodes = 0u;
	u32 cutoffDepthMask = octree.get_root_depth_mask();
	
	currLevel.push_back(octree.get_root_node());
	while(finalNodes < targetVertices && !currLevel.empty()) {
		nextLevel.clear();
		for(const auto& cluster : currLevel) {
			if(!octree.is_leaf(cluster)) {
				const auto children = octree.get_children(cluster);
				for(const auto c : children)
					nextLevel.push_back(c);
			} else {
				octreeNodeMask[cluster.index] = true;
				finalNodes += 1u;
			}
		}
		cutoffDepthMask >>= 1u;
		std::swap(currLevel, nextLevel);
	}

	// TODO: trim if we went over the line
	if(finalNodes > targetVertices) {
		// Reduce cutoff depth by one to indicate that the last level shall not get clustered
		cutoffDepthMask <<= 1u;
	} else {
		// Mark all remaining nodes as end-of-the-line
		for(const auto& cluster : currLevel) {
			octreeNodeMask[cluster.index] = true;
			finalNodes += 1u;
		}
	}
	printf("%llu\n", finalNodes);
	fflush(stdout);

	// We have to track a few things per cluster
	// TODO: better bound!
	std::vector<VertexCluster> clusters(octree.capacity());

	const auto aabbMin = m_boundingBox.min;
	const auto aabbDiag = m_boundingBox.max - m_boundingBox.min;
	// Convenience function to compute the cluster index from a position
	auto get_cluster_index = [&octree, &octreeNodeMask](const ei::Vec3& pos) -> data_structs::CountOctree::NodeId {
		return octree.get_node_id(pos, octreeNodeMask);
	};

	OpenMesh::VPropHandleT<Quadricf> quadricProps{};
	m_meshData->add_property(quadricProps);
	if(!quadricProps.is_valid())
		throw std::runtime_error("Failed to add error quadric property");
	// Compute the error quadrics for each vertex
	// TODO: weight by importance?
	this->compute_error_quadrics(quadricProps);

	// For each vertex, determine the cluster it belongs to and update its statistics
	for(const auto vertex : m_meshData->vertices()) {
		const auto pos = util::pun<ei::Vec3>(m_meshData->point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		if(clusterIndex.levelMask >= cutoffDepthMask)
			clusters[clusterIndex.index].add_vertex(pos, m_meshData->property(quadricProps, vertex));
	}
	m_meshData->remove_property(quadricProps);

	// Calculate the representative cluster position for every cluster
	u32 clusterCount = compute_cluster_center(clusters.begin(), clusters.end());

	// Then we set the position of every vertex to that of its
	// cluster representative
	for(const auto vertex : m_meshData->vertices()) {
		const auto pos = util::pun<ei::Vec3>(m_meshData->point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		// New cluster position has been previously computed
		if(clusterIndex.levelMask >= cutoffDepthMask) {
			const auto newPos = clusters[clusterIndex.index].posAccum;
			m_meshData->point(vertex) = util::pun<PolygonMeshType::Point>(newPos);
		}
	}
	std::vector<PolygonMeshType::HalfedgeHandle> collapses{};
	for(const auto vertex : m_meshData->vertices()) {
		const auto p0 = util::pun<ei::Vec3>(m_meshData->point(vertex));
		for(auto iter = m_meshData->cvoh_begin(vertex); iter.is_valid(); ++iter) {
			const auto vh1 = m_meshData->to_vertex_handle(*iter);
			const auto p1 = util::pun<ei::Vec3>(m_meshData->point(vh1));
			if(p0 == p1)
				collapses.push_back(*iter);
		}
	}
	/*for(const auto heh : collapses) {
		if(m_meshData->is_collapse_ok(heh))
			m_meshData->collapse(heh);
	}*/

	compute_vertex_normals(*m_meshData, m_meshData->vertices_sbegin(), m_meshData->vertices_end());

	if(garbageCollect)
		this->garbage_collect();
	else
		this->rebuild_index_buffer();
	// Do not garbage-collect the mesh yet - only rebuild the index buffer

	// Adjust vertex and face attribute sizes
	m_vertexAttributes.resize(m_meshData->n_vertices());
	m_faceAttributes.resize(m_meshData->n_faces());
	m_vertexAttributes.mark_changed(Device::CPU);
	m_faceAttributes.mark_changed(Device::CPU);

	const auto remainingVertices = std::distance(m_meshData->vertices_sbegin(), m_meshData->vertices_end());
	logInfo("Clustered polygon mesh (", remainingVertices, " vertices, ", clusterCount, " cluster centres)");

	return clusterCount;
}

std::size_t Polygons::cluster(const std::size_t gridRes, bool garbageCollect) {
	using OpenMesh::Geometry::Quadricf;

	this->synchronize<Device::CPU>();

	const auto previousVertices = m_meshData->n_vertices();

	// We have to track a few things per cluster
	std::vector<VertexCluster> clusters(gridRes * gridRes * gridRes);

	const auto aabbMin = m_boundingBox.min;
	const auto aabbDiag = m_boundingBox.max - m_boundingBox.min;
	// Convenience function to compute the cluster index from a position
	auto get_cluster_index = [aabbMin, aabbDiag, gridRes](const ei::Vec3& pos) -> u32 {
		// Get the normalized position [0, 1]^3
		const auto normPos = (pos - aabbMin) / aabbDiag;
		// Get the discretized grid position
		const auto gridPos = ei::min(ei::UVec3{ normPos * ei::Vec3{ gridRes } }, (ei::UVec3{ gridRes } - 1u));
		// Convert the 3D grid position into a 1D index (x -> y -> z)
		const auto gridIndex = gridPos.x + gridPos.y * gridRes + gridPos.z * gridRes * gridRes;
		return static_cast<u32>(gridIndex);
	};
	
	OpenMesh::VPropHandleT<Quadricf> quadricProps{};
	m_meshData->add_property(quadricProps);
	if(!quadricProps.is_valid())
		throw std::runtime_error("Failed to add error quadric property");
	// Compute the error quadrics for each vertex
	this->compute_error_quadrics(quadricProps);

	// For each vertex, determine the cluster it belongs to and update its statistics
	for(const auto vertex : m_meshData->vertices()) {
		const auto pos = util::pun<ei::Vec3>(m_meshData->point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		clusters[clusterIndex].add_vertex(pos, m_meshData->property(quadricProps, vertex));
	}
	m_meshData->remove_property(quadricProps);

	// Calculate the representative cluster position for every cluster
	u32 clusterCount = compute_cluster_center(clusters.begin(), clusters.end());

	// Then we set the position of every vertex to that of its
	// cluster representative
	for(const auto vertex : m_meshData->vertices()) {
		const auto pos = util::pun<ei::Vec3>(m_meshData->point(vertex));
		const auto clusterIndex = get_cluster_index(pos);
		// New cluster position has been previously computed
		const auto newPos = clusters[clusterIndex].posAccum;
		m_meshData->point(vertex) = util::pun<PolygonMeshType::Point>(newPos);
	}
	// After that, we can easily remove degenerate faces (by collapsing degenerate halfedges)
	/*for(const auto edge : m_meshData->edges()) {
		const auto heh = m_meshData->halfedge_handle(edge, 0);
		const auto vh0 = m_meshData->from_vertex_handle(heh);
		const auto vh1 = m_meshData->to_vertex_handle(heh);
		const auto p0 = util::pun<ei::Vec3>(m_meshData->point(vh0));
		const auto p1 = util::pun<ei::Vec3>(m_meshData->point(vh1));
		if((p0 == p1) && m_meshData->is_collapse_ok(heh))
			m_meshData->collapse(heh);
	}*/
	std::vector<PolygonMeshType::HalfedgeHandle> collapses{};
	for(const auto vertex : m_meshData->vertices()) {
		const auto p0 = util::pun<ei::Vec3>(m_meshData->point(vertex));
		for(auto iter = m_meshData->cvoh_begin(vertex); iter.is_valid(); ++iter) {
			const auto vh1 = m_meshData->to_vertex_handle(*iter);
			const auto p1 = util::pun<ei::Vec3>(m_meshData->point(vh1));
			if(p0 == p1)
				collapses.push_back(*iter);
		}
	}
	/*for(const auto heh : collapses) {
		if(m_meshData->is_collapse_ok(heh))
			m_meshData->collapse(heh);
	}*/

	compute_vertex_normals(*m_meshData, m_meshData->vertices_sbegin(), m_meshData->vertices_end());

	if(garbageCollect)
		this->garbage_collect();
	else
		this->rebuild_index_buffer();
	// Do not garbage-collect the mesh yet - only rebuild the index buffer

	// Adjust vertex and face attribute sizes
	m_vertexAttributes.resize(m_meshData->n_vertices());
	m_faceAttributes.resize(m_meshData->n_faces());
	m_vertexAttributes.mark_changed(Device::CPU);
	m_faceAttributes.mark_changed(Device::CPU);

	const auto remainingVertices = std::distance(m_meshData->vertices_sbegin(), m_meshData->vertices_end());
	logInfo("Clustered polygon mesh (", remainingVertices, " vertices, ", clusterCount, " cluster centres)");

	return clusterCount;
}

std::size_t Polygons::decimate(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
							   std::size_t targetVertices, bool garbageCollect) {
	this->synchronize<Device::CPU>();
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
	m_faceAttributes.mark_changed(Device::CPU);
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

void Polygons::garbage_collect(std::function<void(VertexHandle, VertexHandle)> vCallback) {
	if(vCallback) {
		// "Manual" call
		std::vector<PolygonMeshType::VertexHandle*> emptyVh;
		std::vector<PolygonMeshType::HalfedgeHandle*> emptyHh;
		std::vector<PolygonMeshType::FaceHandle*> emptyFh;
		m_meshData->garbage_collection(emptyVh, emptyHh, emptyFh, true, true, true, vCallback);
	} else {
		m_meshData->garbage_collection();
	}
	this->rebuild_index_buffer();
	m_vertexAttributes.shrink_to_fit();
	m_faceAttributes.shrink_to_fit();
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

float Polygons::compute_surface_area() const noexcept {
	float area = 0.f;
	for(const auto face : m_meshData->faces()) {
		auto vIter = m_meshData->cfv_ccwbegin(face);
		const auto a = *vIter; ++vIter;
		const auto b = *vIter; ++vIter;
		const auto c = *vIter; ++vIter;
		const auto pA = util::pun<ei::Vec3>(m_meshData->point(a));
		const auto pB = util::pun<ei::Vec3>(m_meshData->point(b));
		const auto pC = util::pun<ei::Vec3>(m_meshData->point(c));
		area += ei::len(ei::cross(pB - pA, pC - pA));
		if(vIter.is_valid()) {
			const auto d = *vIter;
			const auto pD = util::pun<ei::Vec3>(m_meshData->point(d));
			area += ei::len(ei::cross(pC - pA, pD - pA));
		}
	}
	return 0.5f * area;
}

void Polygons::remove_curvature() {
	if(m_curvatureHdl.has_value()) {
		m_vertexAttributes.remove_attribute(m_curvatureHdl.value());
		m_curvatureHdl.reset();
	}
}

void Polygons::compute_curvature() {
	// Check if the curvature has been computed before
	if(!m_curvatureHdl.has_value())
		m_curvatureHdl = this->template add_vertex_attribute<float>("mean_curvature");

	float* curv = m_vertexAttributes.template acquire<Device::CPU, float>(m_curvatureHdl.value());

	if(!m_meshData->has_vertex_normals()) {
		// Without interpolated normals, the mesh is assumed to have only flat
		// faces -> no curvature.
		for(u32 i = 0; i < m_meshData->n_vertices(); ++i)
			curv[i] = 0.0f;
		return;
	}

	for(auto& v : m_meshData->vertices()) {
		// Fetch data at reference vertex and create an orthogonal space
		Point vPos = ei::details::hard_cast<ei::Vec3>(m_meshData->point(v));
		Direction vNrm = ei::details::hard_cast<ei::Vec3>(m_meshData->normal(v));
		Direction dirU = normalize(perpendicular(vNrm));
		Direction dirV = cross(vNrm, dirU);
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
}


template < Device dev >
void Polygons::synchronize() {
	m_vertexAttributes.synchronize<dev>();
	m_faceAttributes.synchronize<dev>();
	// Synchronize the index buffer

	if(m_indexBuffer.template get<IndexBuffer<dev>>().indices == nullptr) {
		// Try to find a valid device
		bool synced = false;
		m_indexBuffer.for_each([&](auto& buffer) {
			using ChangedBuffer = std::decay_t<decltype(buffer)>;
			if(!synced && buffer.indices != nullptr) {
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

	if(vertexAttribs.size() == 0) {
		if(buffer.vertSize != 0)
			buffer.vertex = Allocator<dev>::template free<ArrayDevHandle_t<dev, void>>(buffer.vertex, buffer.vertSize);
		// else keep current nullptr
	} else {
		if(buffer.vertSize == 0)
			buffer.vertex = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(vertexAttribs.size());
		else
			buffer.vertex = Allocator<dev>::template realloc<ArrayDevHandle_t<dev, void>>(buffer.vertex, buffer.vertSize,
																						  vertexAttribs.size());
	}
	if(faceAttribs.size() == 0) {
		if(buffer.faceSize != 0)
			buffer.face = Allocator<dev>::template free<ArrayDevHandle_t<dev, void>>(buffer.face, buffer.faceSize);
		// else keep current nullptr
	} else {
		if(buffer.faceSize == 0)
			buffer.face = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(faceAttribs.size());
		else
			buffer.face = Allocator<dev>::template realloc<ArrayDevHandle_t<dev, void>>(buffer.face, buffer.faceSize, faceAttribs.size());
	}

	std::vector<ArrayDevHandle_t<dev, void>> cpuVertexAttribs;
	std::vector<ArrayDevHandle_t<dev, void>> cpuFaceAttribs;
	cpuVertexAttribs.reserve(vertexAttribs.size());
	cpuFaceAttribs.reserve(faceAttribs.size());
	for(const auto& ident : vertexAttribs)
		cpuVertexAttribs.push_back(this->template acquire_vertex<dev, void>(ident));
	for(const auto& ident : faceAttribs)
		cpuFaceAttribs.push_back(this->template acquire_face<dev, void>(ident));
	copy(buffer.vertex, cpuVertexAttribs.data(), sizeof(cpuVertexAttribs.front()) * vertexAttribs.size());
	copy(buffer.face, cpuFaceAttribs.data(), sizeof(cpuFaceAttribs.front()) * faceAttribs.size());

	descriptor.numVertexAttributes = static_cast<u32>(vertexAttribs.size());
	descriptor.numFaceAttributes = static_cast<u32>(faceAttribs.size());
	descriptor.vertexAttributes = buffer.vertex;
	descriptor.faceAttributes = buffer.face;
}

// Reserves more space for the index buffer
template < Device dev, bool markChanged >
void Polygons::reserve_index_buffer(std::size_t capacity) {
	auto& buffer = m_indexBuffer.get<IndexBuffer<dev>>();
	if(capacity > buffer.reserved) {
		if(buffer.reserved == 0u)
			buffer.indices = Allocator<dev>::template alloc_array<u32>(capacity);
		else
			buffer.indices = Allocator<dev>::template realloc<u32>(buffer.indices, buffer.reserved,
																   capacity);
		buffer.reserved = capacity;
		if constexpr(markChanged) {
			m_indexBuffer.for_each([](auto& data) {
				using ChangedBuffer = std::decay_t<decltype(data)>;
				if(ChangedBuffer::DEVICE != dev && data.indices != nullptr) {
					Allocator<ChangedBuffer::DEVICE>::template free<u32>(data.indices, data.reserved);
					data.reserved = 0u;
				}
			});
		}
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
		for(auto vertexIter = startVertex; vertexIter != endVertex; ++vertexIter) {
			*(currIndices++) = static_cast<u32>(vertexIter->idx());
			ei::Vec3 pt = util::pun<ei::Vec3>(m_meshData->points()[vertexIter->idx()]);
			m_boundingBox = ei::Box{ m_boundingBox, ei::Box{ util::pun<ei::Vec3>(m_meshData->points()[vertexIter->idx()]) } };
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
		auto& changedBuffer = m_indexBuffer.get<IndexBuffer<changed>>();
		auto& syncBuffer = m_indexBuffer.get<IndexBuffer<sync>>();

		// Check if we need to realloc
		if(syncBuffer.reserved < m_triangles + m_quads)
			this->reserve_index_buffer<sync, false>(3u * m_triangles + 4u * m_quads);

		if(changedBuffer.reserved != 0u)
			copy(syncBuffer.indices, changedBuffer.indices, sizeof(u32) * (3u * m_triangles + 4u * m_quads));
	}
}

template < Device dev >
void Polygons::unload_index_buffer() {
	auto& idxBuffer = m_indexBuffer.template get<IndexBuffer<dev>>();
	if(idxBuffer.indices != nullptr)
		idxBuffer.indices = Allocator<dev>::free(idxBuffer.indices, idxBuffer.reserved);
	idxBuffer.reserved = 0u;
}

template < Device dev >
void Polygons::resizeAttribBuffer(std::size_t v, std::size_t f) {
	AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
	// Resize the attribute array if necessary
	if(attribBuffer.faceSize < f) {
		if(attribBuffer.faceSize == 0)
			attribBuffer.face = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(f);
		else
			attribBuffer.face = Allocator<dev>::template realloc<ArrayDevHandle_t<dev, void>>(attribBuffer.face, attribBuffer.faceSize, f);
		attribBuffer.faceSize = f;
	}
	if(attribBuffer.vertSize < v) {
		if(attribBuffer.vertSize == 0)
			attribBuffer.vertex = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(v);
		else
			attribBuffer.vertex = Allocator<dev>::template realloc< ArrayDevHandle_t<dev, void>>(attribBuffer.vertex, attribBuffer.vertSize, v);
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
template void Polygons::compute_error_quadrics<float>(OpenMesh::VPropHandleT<OpenMesh::Geometry::QuadricT<float>>);


} // namespace mufflon::scene::geometry
