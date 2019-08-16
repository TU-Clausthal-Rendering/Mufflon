#pragma once

#include "polygon_mesh.hpp"
#include "util/assert.hpp"
#include "core/scene/attribute.hpp"
#include "core/scene/types.hpp"
#include "util/range.hpp"
#include <ei/3dtypes.hpp>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <optional>
#include "util/string_view.hpp"
#include <functional>
#include <tuple>
#include <vector>
#include <unordered_set>

// Forward declarations
namespace OpenMesh::Subdivider::Uniform {
template < class Mesh, class Real >
class SubdividerT;
} // namespace OpenMesh::Sudivider::Uniform
namespace OpenMesh::Subdivider::Adaptive {
template < class Mesh >
class CompositeT;
} // namespace OpenMesh::Sudivider::Uniform
namespace OpenMesh::Decimater {
template < class Mesh >
class DecimaterT;
} // namespace OpenMesh::Decimater

namespace mufflon::util {
class IByteReader;
} // namespace mufflon::util

namespace mufflon { namespace scene {

template < Device dev >
struct PolygonsDescriptor;

class Scenario;

namespace tessellation {

class Tessellater;
class TessLevelOracle;

} // namespace tessellation

namespace geometry {

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
			m_faceIter(std::move(iter)) {}

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

	Polygons(const Polygons&);
	Polygons(Polygons&&);
	Polygons& operator=(const Polygons&) = delete;
	Polygons& operator=(Polygons&&) = delete;
	~Polygons();

	void reserve(std::size_t vertices, std::size_t edges, std::size_t tris, std::size_t quads);
	
	template < class T >
	VertexAttributeHandle add_vertex_attribute(std::string name) {
		return m_vertexAttributes.add_attribute<T>(std::move(name));
	}
	template < class T >
	FaceAttributeHandle add_face_attribute(std::string name) {
		return m_faceAttributes.add_attribute<T>(std::move(name));
	}

	bool has_vertex_attribute(StringView name) {
		return m_vertexAttributes.has_attribute(name);
	}
	bool has_face_attribute(StringView name) {
		return m_faceAttributes.has_attribute(name);
	}

	void remove_attribute(StringView name) {
		throw std::runtime_error("Operation not implemented yet");
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize();

	template < Device dev >
	void synchronize(VertexAttributeHandle hdl) {
		m_vertexAttributes.synchronize<dev>(hdl);
	}
	template < Device dev >
	void synchronize(FaceAttributeHandle hdl) {
		m_faceAttributes.synchronize<dev>(hdl);
	}
	template < Device dev, bool face >
	void synchronize(StringView name) {
		get_attributes<face>().synchronize<dev>(name);
	}

	template < Device dev >
	void unload() {
		m_vertexAttributes.unload<dev>();
		m_faceAttributes.unload<dev>();
		unload_index_buffer<dev>();
	}

	template < bool face >
	void mark_changed(Device dev) {
		get_attributes<face>().mark_changed(dev);
	}
	void mark_changed(Device dev);

	// Gets the descriptor with only default attributes (position etc)
	template < Device dev >
	PolygonsDescriptor<dev> get_descriptor();
	// Updates the descriptor with the given set of attributes
	template < Device dev >
	void update_attribute_descriptor(PolygonsDescriptor<dev>& descriptor,
									 const std::vector<const char*>& vertexAttribs,
									 const std::vector<const char*>& faceAttribs);

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
	std::size_t add_bulk(StringView name, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(StringView name, const FaceHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(VertexAttributeHandle hdl, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(FaceAttributeHandle hdl, const FaceHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);

	// Implements tessellation for the mesh
	void tessellate(tessellation::Tessellater& tessellater);
	// Implements displacement mapping for the mesh
	void displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario);

	// Creates a decimater 
	OpenMesh::Decimater::DecimaterT<PolygonMeshType> create_decimater();
	// Implements decimation.
	std::size_t decimate(OpenMesh::Decimater::DecimaterT<PolygonMeshType>& decimater,
						 std::size_t targetVertices, bool garbageCollect);

	// Splits a vertex
	std::pair<FaceHandle, FaceHandle> vertex_split(const VertexHandle v0, const VertexHandle v1,
												   const VertexHandle vl, const VertexHandle vr);
	// Garbage-collects the mesh and the index buffer
	void garbage_collect(std::function<void(VertexHandle, VertexHandle)> vCallback = {});

	// Transforms polygon data
	void transform(const ei::Mat3x4& transMat, const ei::Vec3& scale);

	// Computes the "mean_curvature" attribute for all vertices
	void compute_curvature();

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

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(VertexAttributeHandle hdl) {
		return m_vertexAttributes.acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(FaceAttributeHandle hdl) {
		return m_faceAttributes.acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(VertexAttributeHandle hdl) {
		return m_vertexAttributes.acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(FaceAttributeHandle hdl) {
		return m_faceAttributes.acquire_const<dev, T>(hdl);
	}
	template < Device dev, class T, bool face >
	ArrayDevHandle_t<dev, T> acquire(StringView name) {
		return get_attributes<face>().acquire<dev, T>(name);
	}
	template < Device dev, class T, bool face >
	ConstArrayDevHandle_t<dev, T> acquire_const(StringView name) {
		return get_attributes<face>().acquire_const<dev, T>(name);
	}

	VertexAttributeHandle get_points_hdl() const noexcept {
		return m_pointsHdl;
	}
	VertexAttributeHandle get_normals_hdl() const noexcept {
		return m_normalsHdl;
	}
	VertexAttributeHandle get_uvs_hdl() const noexcept {
		return m_uvsHdl;
	}
	FaceAttributeHandle get_material_indices_hdl() const noexcept {
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

	// Get a list of all materials which are referenced by any primitive
	const std::unordered_set<MaterialIndex>& get_unique_materials() const {
		return m_uniqueMaterials;
	}

	PolygonMeshType& get_mesh() noexcept {
		return *m_meshData;
	}

	const PolygonMeshType& get_mesh() const noexcept {
		return *m_meshData;
	}

	// Returns whether any polygon has a displacement map associated with the given material assignment
	bool has_displacement_mapping(const Scenario& scenario) const noexcept;

	bool was_displacement_mapping_applied() const noexcept {
		return m_wasDisplaced;
	}

private:
	static constexpr const char MAT_INDICES_NAME[] = "material-indices";

	template < Device dev >
	struct DescFlags {
		bool geometryChanged;
	};

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

	using IndexBuffers = util::TaggedTuple<
		IndexBuffer<Device::CPU>,
		IndexBuffer<Device::CUDA>,
		IndexBuffer<Device::OPENGL>
	>;
	using AttribBuffers = util::TaggedTuple<
		AttribBuffer<Device::CPU>,
		AttribBuffer<Device::CUDA>,
		AttribBuffer<Device::OPENGL>
	>;

	// Helper for deciding between vertex and face attributes
	template < bool face >
	auto& get_attributes() {
		if constexpr(face)
			return m_faceAttributes;
		else
			return m_vertexAttributes;
	}

	// Reserves more space for the index buffer
	template < Device dev, bool markChanged = true >
	void reserve_index_buffer(std::size_t capacity);
	// Rebuilds the index buffer from scratch
	void rebuild_index_buffer();
	// Synchronizes two device index buffers
	template < Device changed, Device sync >
	void synchronize_index_buffer();
	template < Device dev >
	void unload_index_buffer();
	// Resizes the attribute buffer to hold v vertex and f face attribute pointers
	template < Device dev >
	void resizeAttribBuffer(std::size_t v, std::size_t f);

	// It's a unique pointer so we have one fixed address we can reference in OmAttributePool
	// TODO: does that degrade performance? probably not, since attributes aren't aquired often
	std::unique_ptr<PolygonMeshType> m_meshData;
	VertexAttributePoolType m_vertexAttributes;
	FaceAttributePoolType m_faceAttributes;
	VertexAttributeHandle m_pointsHdl;
	VertexAttributeHandle m_normalsHdl;
	VertexAttributeHandle m_uvsHdl;
	FaceAttributeHandle m_matIndicesHdl;
	// Vertex-index buffer, first for the triangles, then for quads
	IndexBuffers m_indexBuffer;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;

	// Dirty flags for descriptor rebuilding
	util::TaggedTuple<DescFlags<Device::CPU>, DescFlags<Device::CUDA>> m_descFlags;

	ei::Box m_boundingBox;
	std::size_t m_triangles = 0u;
	std::size_t m_quads = 0u;

	// Whenever a primitive is added the table of all referenced
	// materials will be updated. Assumption: a material reference
	// will not change afterwards.
	std::unordered_set<MaterialIndex> m_uniqueMaterials;

	// Keeps track of whether displacement mapping was already applied or not
	bool m_wasDisplaced = false;
};

}}} // namespace mufflon::scene::geometry
