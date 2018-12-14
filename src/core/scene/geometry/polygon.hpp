#pragma once

#include "polygon_mesh.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "core/scene/attr.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "util/range.hpp"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <array>
#include <optional>
#include <string_view>
#include <tuple>

// Forward declarations
namespace OpenMesh::Subdivider::Uniform {
template < class Mesh, class Real >
class SubdividerT;
} // namespace OpenMesh::Sudivider::Uniform
namespace OpenMesh::Subdivider::Adaptive {
template < class Mesh >
class CompositeT;
} // namespace OpenMesh::Sudivider::Uniform
namespace OpenMesh::Decimater{
template < class Mesh >
class DecimaterT;
} // namespace OpenMesh::Decimater

namespace mufflon::util {
class IByteReader;
} // namespace mufflon::util

namespace mufflon::scene::geometry {

/**
 * Instantiation of geometry class.
 * Can store both triangles and quads.
 * Can be extended to work with any polygon type.
 */
class Polygons {
public:
	// Basic type definitions
	using Index = u32;
	using Triangle = std::array<Index, 3u>;
	using Quad = std::array<Index, 4u>;
	// OpenMesh types
	// TODO: change attributelist
	using VertexAttributePoolType = OpenMeshAttributePool<false>;
	using FaceAttributePoolType = OpenMeshAttributePool<true>;
	using VertexHandle = OpenMesh::VertexHandle;
	using FaceHandle = OpenMesh::FaceHandle;
	using TriangleHandle = OpenMesh::FaceHandle;
	using QuadHandle = OpenMesh::FaceHandle;

	// Struct for communicating the number of bulk-read vertex attributes
	struct VertexBulkReturn {
		VertexHandle handle;
		std::size_t readPoints;
		std::size_t readNormals;
		std::size_t readUvs;
	};

	// Associates an attribute name with a type (vertex- or faceattributehandle)
	template < class T >
	struct VAttrDesc {
		using Type = T;
		std::string name;
	};
	template < class T >
	struct FAttrDesc {
		using Type = T;
		std::string name;
	};
	
	class FaceIterator {
	public:
		static FaceIterator cbegin(const PolygonMeshType& mesh) {
			return FaceIterator(mesh, mesh.faces().begin());
		}

		static FaceIterator cend(const PolygonMeshType& mesh) {
			return FaceIterator(mesh, mesh.faces().end());
		}

		FaceIterator& operator++() {
			++m_faceIter;
			return *this;
		}

		FaceIterator operator++(int) {
			FaceIterator temp(*this);
			++(*this);
			return temp;
		}

		bool operator!=(const FaceIterator &other) {
			return m_faceIter != other.m_faceIter;
		}

		std::size_t get_vertex_count() const {
			mAssert(m_faceIter != m_mesh.faces().end());
			mAssert(m_faceIter->is_valid());
			mAssert(static_cast<std::size_t>(m_faceIter->idx()) < m_mesh.n_faces());
			return std::distance(m_mesh.cfv_ccwbegin(*m_faceIter), m_mesh.cfv_ccwend(*m_faceIter));
		}

		OpenMesh::PolyConnectivity::ConstFaceVertexRange operator*() const {
			mAssert(m_faceIter != m_mesh.faces().end());
			mAssert(m_faceIter->is_valid());
			mAssert(static_cast<std::size_t>(m_faceIter->idx()) < m_mesh.n_faces());
			return m_mesh.fv_range(*m_faceIter);
		}

		const OpenMesh::PolyConnectivity::ConstFaceIter& operator->() const noexcept {
			return m_faceIter;
		}

	private:
		FaceIterator(const PolygonMeshType& mesh, OpenMesh::PolyConnectivity::ConstFaceIter iter) :
			m_mesh(mesh),
			m_faceIter(std::move(iter))
		{}

		const PolygonMeshType& m_mesh;
		OpenMesh::PolyConnectivity::ConstFaceIter m_faceIter;
	};

	// Ensure matching data types
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Point)
				  && sizeof(Point) == 3u * sizeof(float)
				  && alignof(OpenMesh::Vec3f) == alignof(Point),
				  "Point type must be compatible to OpenMesh");
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Normal)
				  && sizeof(Normal) == 3u * sizeof(float)
				  && alignof(OpenMesh::Vec3f) == alignof(Normal),
				  "Normal type must be compatible to OpenMesh");

	// Default construction, creates material-index attribute.
	Polygons();
	// Creates polygon from already-created mesh.
	Polygons(PolygonMeshType&& mesh);

	Polygons(const Polygons&) = delete;
	Polygons(Polygons&&);
	Polygons& operator=(const Polygons&) = delete;
	Polygons& operator=(Polygons&&) = delete;
	~Polygons();

	void resize(std::size_t vertices, std::size_t edges, std::size_t tris, std::size_t quads);
	void reserve(std::size_t vertices, std::size_t egdes, std::size_t tris, std::size_t quads);

	template < class T, bool face >
	typename OpenMeshAttributePool<face>::AttributeHandle add_attribute(std::string name) {
		return get_attributes<face>().add_attribute<T>(std::move(name));
	}

	void remove_attribute(std::string_view name) {
		throw std::runtime_error("Operation not implemented yet");
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_vertexAttributes.synchronize<dev>();
		m_faceAttributes.synchronize<dev>();
		// Synchronize the index buffer
		if (m_indexFlags.needs_sync(dev) && m_indexFlags.has_changes()) {
			if (m_indexFlags.has_competing_changes()) {
				logError("[Polygons::synchronize] Competing device changes; ignoring one");
			}
			m_indexBuffer.for_each([&](auto& buffer) {
				using ChangedBuffer = std::decay_t<decltype(buffer)>;
				this->synchronize_index_buffer<ChangedBuffer::DEVICE, dev>();
			});
		}
	}

	template < Device dev, bool face >
	void synchronize(typename OpenMeshAttributePool<face>::AttributeHandle hdl) {
		get_attributes<face>().synchronize<dev>(hdl);
	}
	template < Device dev, bool face >
	void synchronize(std::string_view name) {
		get_attributes<face>().synchronize<dev>(name);
	}

	template < Device dev >
	void unload() {
		m_vertexAttributes.unload<dev>();
		m_faceAttributes.unload<dev>();
	}

	template < bool face >
	void mark_changed(Device dev) {
		get_attributes<face>().mark_changed(dev);
	}
	template < bool face >
	void mark_changed(Device dev, typename OpenMeshAttributePool<face>::AttributeHandle hdl) {
		get_attributes<face>().mark_changed(dev, hdl);
	}
	template < bool face >
	void mark_changed(Device dev, std::string_view name) {
		get_attributes<face>().mark_changed(dev, name);
	}

	/**
	 * Returns a descriptor (on CPU side) with pointers to resources (on Device side).
	 * Takes two tuples: they must each contain the name and type of attributes which the
	 * renderer wants to have access to. If an attribute gets written to, it is the
	 * renderer's task to aquire it once more after that, since we cannot hand out
	 * Accessors to the concrete device.
	 */
	template < Device dev, std::size_t N, std::size_t M >
	PolygonsDescriptor<dev> get_descriptor(const std::array<const char*, N>& vertexAttribs,
										   const std::array<const char*, M>& faceAttribs) {
		this->synchronize<dev>();

		// Resize the attribute array if necessary
		resizeAttribBuffer<dev>(vertexAttribs.size(), faceAttribs.size());
		// Collect the attributes; for that, we iterate the given Attributes and
		// gather them on CPU side (or rather, their device pointers); then
		// we copy it to the actual device
		AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
		std::vector<void*> cpuVertexAttribs(vertexAttribs.size());
		std::vector<void*> cpuFaceAttribs(faceAttribs.size());
		for (const char* name : vertexAttribs)
			cpuVertexAttribs.push_back(m_vertexAttributes.acquire<dev, void>(name));
		for (const char* name : faceAttribs)
			cpuFaceAttribs.push_back(m_faceAttributes.acquire<dev, void>(name));
		copy<void*>(attribBuffer.vertex, cpuVertexAttribs.data(), vertexAttribs.size());
		copy<void*>(attribBuffer.face, cpuFaceAttribs.data(), faceAttribs.size());

		return PolygonsDescriptor<dev>{
			static_cast<u32>(this->get_vertex_count()),
			static_cast<u32>(this->get_triangle_count()),
			static_cast<u32>(this->get_quad_count()),
			static_cast<u32>(vertexAttribs.size()),
			static_cast<u32>(faceAttribs.size()),
			this->acquire_const<dev, ei::Vec3, false>(this->get_points_hdl()),
			this->acquire_const<dev, ei::Vec3, false>(this->get_normals_hdl()),
			this->acquire_const<dev, ei::Vec2, false>(this->get_uvs_hdl()),
			this->acquire_const<dev, u16, true>(this->get_material_indices_hdl()),
			this->get_index_buffer<dev>(),
			attribBuffer.vertex,
			attribBuffer.face
		};
	}

	// Adds a new vertex.
	VertexHandle add(const Point& point, const Normal& normal, const UvCoordinate& uv);
	// Adds a new triangle.
	TriangleHandle add(const Triangle& tri);
	TriangleHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2);
	TriangleHandle add(const std::array<VertexHandle, 3u>& vertices);
	TriangleHandle add(const Triangle& tri, MaterialIndex idx);
	TriangleHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
					   MaterialIndex idx);
	TriangleHandle add(const std::array<VertexHandle, 3u>& vertices, MaterialIndex idx);
	// Adds a new quad.
	QuadHandle add(const Quad& quad);
	QuadHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
				   const VertexHandle& v3);
	QuadHandle add(const std::array<VertexHandle, 4u>& vertices);
	QuadHandle add(const Quad& quad, MaterialIndex idx);
	QuadHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
				   const VertexHandle& v3, MaterialIndex idx);
	QuadHandle add(const std::array<VertexHandle, 4u>& vertices, MaterialIndex idx);

	/**
	 * Adds a bulk of vertices.
	 * Returns both a handle to the first added vertex as well as the number of
	 * read vertices.
	 */
	VertexBulkReturn add_bulk(std::size_t count, util::IByteReader& pointStream,
							  util::IByteReader& normalStream, util::IByteReader& uvStream);
	VertexBulkReturn add_bulk(std::size_t count, util::IByteReader& pointStream,
							  util::IByteReader& normalStream, util::IByteReader& uvStream,
							  const ei::Box& boundingBox);
	VertexBulkReturn add_bulk(std::size_t count, util::IByteReader& pointStream,
							  util::IByteReader& uvStream);
	VertexBulkReturn add_bulk(std::size_t count, util::IByteReader& pointStream,
							  util::IByteReader& uvStream, const ei::Box& boundingBox);
	/**
	 * Bulk-loads the given attribute starting at the given vertex/face.
	 * The number of read values will be capped by the number of vertice present
	 * after the starting position.
	 */
	std::size_t add_bulk(std::string_view name, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(std::string_view name, const FaceHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(OpenMeshAttributePool<false>::AttributeHandle hdl, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(OpenMeshAttributePool<true>::AttributeHandle hdl, const FaceHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);

	// Implements tessellation for uniform subdivision.
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<PolygonMeshType, Real>& tessellater,
					std::size_t divisions);
	// Implements tessellation for adaptive subdivision.
	/*void tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<PolygonMeshType>& tessellater,
					std::size_t divisions);*/
	// Implements decimation.
	void create_lod(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
					std::size_t target_vertices);

	// Gets a constant handle to the underlying mesh data.
	const PolygonMeshType& native() const {
		mAssert(m_meshData != nullptr);
		return *m_meshData;
	}

	// Get iterator over all faces (and vertices for the faces)
	util::Range<FaceIterator> faces() const {
		mAssert(m_meshData != nullptr);
		return util::Range<FaceIterator>{
			FaceIterator::cbegin(*m_meshData),
			FaceIterator::cend(*m_meshData)
		};
	}

	template < Device dev >
	ConstArrayDevHandle_t<dev, u32> get_index_buffer() {
		return ConstArrayDevHandle_t<dev, u32>{
			m_indexBuffer.get<IndexBuffer<dev>>().indices
		};
	}

	template < Device dev, class T, bool face >
	T* acquire(typename OpenMeshAttributePool<face>::AttributeHandle hdl) {
		return get_attributes<face>().acquire<dev, T>(hdl);
	}
	template < Device dev, class T, bool face >
	const T* acquire_const(typename OpenMeshAttributePool<face>::AttributeHandle hdl) {
		return get_attributes<face>().acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T, bool face >
	T* acquire(std::string_view name) {
		return get_attributes<face>().acquire<dev, T>(name);
	}
	template < Device dev, class T, bool face >
	const T* acquire_const(std::string_view name) {
		return get_attributes<face>().acquire_const<dev, T>(name);
	}

	OpenMeshAttributePool<false>::AttributeHandle get_points_hdl() const noexcept {
		return m_pointsHdl;
	}
	OpenMeshAttributePool<false>::AttributeHandle get_normals_hdl() const noexcept {
		return m_normalsHdl;
	}
	OpenMeshAttributePool<false>::AttributeHandle get_uvs_hdl() const noexcept {
		return m_uvsHdl;
	}
	OpenMeshAttributePool<true>::AttributeHandle get_material_indices_hdl() const noexcept {
		return m_matIndicesHdl;
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_vertex_count() const noexcept {
		return m_meshData->n_vertices();
	}

	std::size_t get_edge_count() const noexcept {
		return m_meshData->n_edges();
	}

	std::size_t get_triangle_count() const noexcept {
		return m_triangles;
	}

	std::size_t get_quad_count() const noexcept {
		return m_quads;
	}

	std::size_t get_face_count() const noexcept {
		return m_meshData->n_faces();
	}

	std::size_t get_vertex_attribute_count() const noexcept {
		return m_vertexAttributes.get_attribute_count();
	}

	std::size_t get_face_attribute_count() const noexcept {
		return m_faceAttributes.get_attribute_count();
	}

private:
	static constexpr const char MAT_INDICES_NAME[] = "material-indices";

	// Helper class for distinct array handle types
	template < Device dev >
	struct IndexBuffer {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, u32> indices;
		std::size_t reserved = 0u;
	};
	template < Device dev >
	struct AttribBuffer {
		static constexpr Device DEVICE = dev;
		ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> vertex;
		ArrayDevHandle_t<dev, ArrayDevHandle_t<dev, void>> face;
		std::size_t vertSize = 0u;
		std::size_t faceSize = 0u;
	};

	using IndexBuffers = util::TaggedTuple<IndexBuffer<Device::CPU>,
		IndexBuffer<Device::CUDA>>;
	using AttribBuffers = util::TaggedTuple<AttribBuffer<Device::CPU>,
		AttribBuffer<Device::CUDA>>;

	// Helper for deciding between vertex and face attributes
	template < bool face >
	auto& get_attributes() {
		if constexpr(face)
			return m_faceAttributes;
		else
			return m_vertexAttributes;
	}

	// Reserves more space for the index buffer
	template < Device dev >
	void reserve_index_buffer(std::size_t capacity) {
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
	void synchronize_index_buffer() {
		if constexpr(changed != sync) {
			if(m_indexFlags.has_changes(changed)) {
				auto& changedBuffer = m_indexBuffer.get<IndexBuffer<changed>>();
				auto& syncBuffer = m_indexBuffer.get<IndexBuffer<sync>>();

				// Check if we need to realloc
				if(syncBuffer.reserved < m_triangles + m_quads)
					this->reserve_index_buffer<sync>(m_triangles + m_quads);

				if(changedBuffer.reserved != 0u)
					copy(syncBuffer.indices, changedBuffer.indices, m_triangles + m_quads);
				m_indexFlags.mark_synced(sync);
			}
		}
	}

	template < Device dev >
	void resizeAttribBuffer(std::size_t v, std::size_t f) {
		AttribBuffer<dev>& attribBuffer = m_attribBuffer.get<AttribBuffer<dev>>();
		// Resize the attribute array if necessary
		if (attribBuffer.faceSize < f) {
			if (attribBuffer.faceSize == 0)
				attribBuffer.face = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(f);
			else
				attribBuffer.face = Allocator<dev>::realloc(attribBuffer.face, attribBuffer.faceSize, f);
			attribBuffer.faceSize = f;
		}
		if (attribBuffer.vertSize < v) {
			if (attribBuffer.vertSize == 0)
				attribBuffer.vertex = Allocator<dev>::template alloc_array<ArrayDevHandle_t<dev, void>>(v);
			else
				attribBuffer.vertex = Allocator<dev>::realloc(attribBuffer.vertex, attribBuffer.vertSize, v);
			attribBuffer.vertSize = v;
		}
	}

	// It's a unique pointer so we have one fixed address we can reference in OmAttributePool
	// TODO: does that degrade performance? probably not, since attributes aren't aquired often
	std::unique_ptr<PolygonMeshType> m_meshData;
	VertexAttributePoolType m_vertexAttributes;
	FaceAttributePoolType m_faceAttributes;
	OpenMeshAttributePool<false>::AttributeHandle m_pointsHdl;
	OpenMeshAttributePool<false>::AttributeHandle m_normalsHdl;
	OpenMeshAttributePool<false>::AttributeHandle m_uvsHdl;
	OpenMeshAttributePool<true>::AttributeHandle m_matIndicesHdl;
	// Vertex-index buffer, first for the triangles, then for quads
	IndexBuffers m_indexBuffer;
	util::DirtyFlags<Device> m_indexFlags;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;

	ei::Box m_boundingBox;
	std::size_t m_triangles = 0u;
	std::size_t m_quads = 0u;
};

} // namespace mufflon::scene::geometry