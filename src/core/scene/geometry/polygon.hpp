#pragma once

#include "polygon_mesh.hpp"
#include "util/range.hpp"
#include "util/string_view.hpp"
#include "core/memory/unique_device_ptr.hpp"
#include "core/renderer/decimaters/util/octree.hpp"
#include "core/scene/types.hpp"
#include "core/scene/attributes/attribute.hpp"
#include "core/scene/attributes/attribute_handles.hpp"
#include <ei/3dtypes.hpp>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <atomic>
#include <functional>
#include <optional>
#include <tuple>
#include <vector>

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
namespace OpenMesh::Geometry {
template < class >
class QuadricT;
}

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
	using VertexAttributePoolType = AttributePool;
	using FaceAttributePoolType = AttributePool;
	using VertexHandle = u32;
	using FaceHandle = u32;
	using TriangleHandle = u32;
	using QuadHandle = u32;

	// Struct for communicating the number of bulk-read vertex attributes
	struct VertexBulkReturn {
		VertexHandle handle;
		std::size_t readPoints;
		std::size_t readNormals;
		std::size_t readUvs;
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

	// Triangle/Quad iterators
	class TriangleIter {
	public:
		ei::UVec3 operator*() const noexcept {
			return ei::UVec3{ m_indices[m_currIndex], m_indices[m_currIndex + 1u], m_indices[m_currIndex + 2u] };
		}
		TriangleIter& operator++() noexcept { m_currIndex += 3u; return *this; }
		TriangleIter operator++(int) noexcept { TriangleIter temp{ *this }; m_currIndex += 3u; return temp; }
		bool operator==(const TriangleIter& rhs) const noexcept {
			return (m_indices == rhs.m_indices) && (m_currIndex == rhs.m_currIndex);
		}
		bool operator!=(const TriangleIter& rhs) const noexcept { return !((*this) == rhs); }

	private:
		friend class Polygons;

		TriangleIter(const u32* indices, std::size_t currIndex) :
			m_indices{ indices },
			m_currIndex{ currIndex }
		{}

		const u32* m_indices;
		std::size_t m_currIndex;
	};
	class QuadIter {
	public:
		ei::UVec4 operator*() const noexcept {
			return ei::UVec4{ m_indices[m_currIndex], m_indices[m_currIndex + 1u],
							  m_indices[m_currIndex + 2u], m_indices[m_currIndex + 3u] };
		}
		QuadIter& operator++() noexcept { m_currIndex += 4u; return *this; }
		QuadIter operator++(int) noexcept { QuadIter temp{ *this }; m_currIndex += 3u; return temp; }

		bool operator==(const QuadIter& rhs) const noexcept {
			return (m_indices == rhs.m_indices) && (m_currIndex == rhs.m_currIndex);
		}
		bool operator!=(const QuadIter& rhs) const noexcept { return !((*this) == rhs); }

	private:
		friend class Polygons;

		QuadIter(const u32* indices, std::size_t currIndex) :
			m_indices{ indices },
			m_currIndex{ currIndex }
		{}

		const u32* m_indices;
		std::size_t m_currIndex;
	};

	// Default construction, creates material-index attribute.
	Polygons();
	// Create a polygon from an existing mesh
	Polygons(const PolygonMeshType& mesh, const OpenMesh::FPropHandleT<MaterialIndex> mats = OpenMesh::FPropHandleT<MaterialIndex>{});

	Polygons(const Polygons&);
	Polygons(Polygons&&);
	Polygons& operator=(const Polygons&) = delete;
	Polygons& operator=(Polygons&&) = delete;
	~Polygons() = default;

	void reserve(std::size_t vertices, std::size_t tris, std::size_t quads);

	VertexAttributeHandle add_vertex_attribute(StringView name, AttributeType type) {
		// Special casing to accelerate animation query
		if(name == StringView("AnimationWeights")) {
			m_animationWeightHdl = m_vertexAttributes.add_attribute(AttributeIdentifier{ type, name });
			return m_animationWeightHdl.value();
		}

		return m_vertexAttributes.add_attribute(AttributeIdentifier{ type, name });
	}
	FaceAttributeHandle add_face_attribute(StringView name, AttributeType type) {
		return m_faceAttributes.add_attribute(AttributeIdentifier{ type, name });
	}
	template < class T >
	VertexAttributeHandle add_vertex_attribute(StringView name) {
		return m_vertexAttributes.add_attribute(AttributeIdentifier{ get_attribute_type<T>(), name });
	}
	template < class T >
	FaceAttributeHandle add_face_attribute(StringView name) {
		return m_faceAttributes.add_attribute(AttributeIdentifier{ get_attribute_type<T>(), name });
	}

	std::optional<VertexAttributeHandle> find_vertex_attribute(StringView name, const AttributeType& type) const {
		const AttributeIdentifier ident{ type, name };
		return m_vertexAttributes.find_attribute(ident);
	}
	std::optional<FaceAttributeHandle> find_face_attribute(StringView name, const AttributeType& type) const {
		const AttributeIdentifier ident{ type, name };
		return m_faceAttributes.find_attribute(ident);
	}
	
	template < class T >
	void remove_vertex_attribute(const std::string& name) {
		const AttributeIdentifier ident{ get_attribute_type<T>(), name };
		if(const auto handle = m_vertexAttributes.find_attribute(ident); handle.has_value())
			m_vertexAttributes.remove_attribute(handle.value());
	}
	template < class T >
	void remove_face_attribute(const std::string& name) {
		const AttributeIdentifier ident{ get_attribute_type<T>(), name };
		if(const auto handle = m_faceAttributes.find_attribute(ident); handle.has_value())
			m_faceAttributes.remove_attribute(handle.value());
	}
	void remove_curvature();

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize();

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
	// Gets the size of the final descriptor
	std::size_t desciptor_size() const noexcept {
		return this->get_vertex_count() * (2u * sizeof(ei::Vec3) +  sizeof(ei::Vec2))
			+ this->get_triangle_count() * (3u * sizeof(u32) + sizeof(MaterialIndex))
			+ this->get_quad_count() * (4u * sizeof(u32) + sizeof(MaterialIndex));
	}
	// Updates the descriptor with the given set of attributes
	template < Device dev >
	void update_attribute_descriptor(PolygonsDescriptor<dev>& descriptor,
									 const std::vector<AttributeIdentifier>& vertexAttribs,
									 const std::vector<AttributeIdentifier>& faceAttribs);

	// Adds a new vertex.
	VertexHandle add(const Point& point, const Normal& normal, const UvCoordinate& uv);
	// Adds a new triangle.
	TriangleHandle add(const Triangle& tri);
	TriangleHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2);
	TriangleHandle add(const Triangle& tri, MaterialIndex idx);
	TriangleHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
					   MaterialIndex idx);
	// Adds a new quad.
	QuadHandle add(const Quad& quad);
	QuadHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
				   const VertexHandle& v3);
	QuadHandle add(const Quad& quad, MaterialIndex idx);
	QuadHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
				   const VertexHandle& v3, MaterialIndex idx);

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
	std::size_t add_bulk(const VertexAttributeHandle& hdl, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream);
	std::size_t add_bulk(const FaceAttributeHandle& hdl, const FaceHandle& startFace,
						 std::size_t count, util::IByteReader& attrStream);

	// TODO: add bulk triangle/quad

	/**
	 * To perform mesh operations that require neighborhood information, index buffers are not enough.
	 * For that we (temporarily!) construct an OpenMesh mesh instance, which encapsulates a half-edge
	 * data structure. To match the attributes etc. you can write it back into the polygon.
	 */
	PolygonMeshType create_halfedge_structure();
	void create_halfedge_structure(PolygonMeshType& mesh);

	// Reconstructs the attribute and index data from the given mesh.
	// Faces/vertices/edges may have been removed from the mesh, but not added.
	// Garbage collection must not have been called yet.
	void reconstruct_from_reduced_mesh(const PolygonMeshType& mesh,
									   std::vector<u32>* newVertexPosition = nullptr,
									   std::vector<ei::Vec3>* normals = nullptr);

	void cluster_uniformly(const ei::UVec3& gridRes);

	// Implements tessellation for the mesh
	void tessellate(tessellation::TessLevelOracle& oracle, const Scenario* scenario,
					const bool usePhong);
	// Implements displacement mapping for the mesh
	void displace(tessellation::TessLevelOracle& oracle, const Scenario& scenario);
	// Checks if the polygon has a bone animation
	bool has_bone_animation() const noexcept {
		return m_animationWeightHdl.has_value();
	}

	// Apply bone animation transformations if this mesh has animation weights
	// Returns wether there was an animation or not (in which case nothing was done).
	bool apply_animation(u32 frame, const Bone* bones);

	// Transforms polygon data
	void transform(const ei::Mat3x4& transMat);

	float compute_surface_area();

	// Computes the "mean_curvature" attribute for all vertices
	void compute_curvature();

	template < Device dev >
	ConstArrayDevHandle_t<dev, u32> get_index_buffer() {
		return ConstArrayDevHandle_t<dev, u32>{ m_indexBuffer.get<IndexBuffer<dev>>().indices.get() };
	}

	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(const VertexAttributeHandle& hdl) {
		return m_vertexAttributes.template acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire(const FaceAttributeHandle& hdl) {
		return m_faceAttributes.template acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(const VertexAttributeHandle& hdl) {
		return this->template acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const(const FaceAttributeHandle& hdl) {
		return this->template acquire<dev, T>(hdl);
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire_vertex(const AttributeIdentifier& ident) {
		if(const auto handle = m_vertexAttributes.find_attribute(ident); handle.has_value())
			return this->template acquire<dev, T>(VertexAttributeHandle{ handle.value() });
		return {};
	}
	template < Device dev, class T >
	ArrayDevHandle_t<dev, T> acquire_face(const AttributeIdentifier& ident) {
		if(const auto handle = m_faceAttributes.find_attribute(ident); handle.has_value())
			return this->template acquire<dev, T>(FaceAttributeHandle{ handle.value() });
		return {};
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const_vertex(const AttributeIdentifier& ident) {
		return this->template acquire_vertex<dev, T>(ident);
	}
	template < Device dev, class T >
	ConstArrayDevHandle_t<dev, T> acquire_const_face(const AttributeIdentifier& ident) {
		return this->template acquire_face<dev, T>(ident);
	}

	const VertexAttributeHandle& get_points_hdl() const noexcept {
		return m_pointsHdl;
	}
	const VertexAttributeHandle& get_normals_hdl() const noexcept {
		return m_normalsHdl;
	}
	const VertexAttributeHandle& get_uvs_hdl() const noexcept {
		return m_uvsHdl;
	}
	const FaceAttributeHandle& get_material_indices_hdl() const noexcept {
		return m_matIndicesHdl;
	}
	std::optional<VertexAttributeHandle> get_curvature_hdl() const noexcept {
		return m_curvatureHdl;
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_vertex_count() const noexcept {
		return m_vertexAttributes.get_attribute_elem_count();
	}

	std::size_t get_triangle_count() const noexcept {
		return m_triangles;
	}

	std::size_t get_quad_count() const noexcept {
		return m_quads;
	}

	std::size_t get_face_count() const noexcept {
		return get_triangle_count() + get_quad_count();
	}

	util::Range<TriangleIter> triangles() {
		const auto indices = this->template get_index_buffer<Device::CPU>();
		return util::Range{ TriangleIter{ indices, 0u }, TriangleIter{ indices, 3u * m_triangles } };
	}
	util::Range<QuadIter> quads() {
		const auto indices = this->template get_index_buffer<Device::CPU>() + 3u * m_triangles;
		return util::Range{ QuadIter{ indices, 0u }, QuadIter{ indices, 4u * m_quads} };
	}

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
		unique_device_ptr<dev, u32[]> indices;
		std::size_t reserved = 0u;
	};
	template < Device dev >
	struct AttribBuffer {
		static constexpr Device DEVICE = dev;
		unique_device_ptr<dev, ArrayDevHandle_t<dev, void>[]> vertex;
		unique_device_ptr<dev, ArrayDevHandle_t<dev, void>[]> face;
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

	// Step after tessellation/displacement which adds the new faces
	void after_tessellation(const PolygonMeshType& mesh, const OpenMesh::FaceHandle tempHandle,
							const OpenMesh::FPropHandleT<OpenMesh::FaceHandle>& oldFaceProp);
	void rebuild_index_buffer(const PolygonMeshType& mesh, const OpenMesh::FaceHandle tempHandle);
	// Reserves more space for the index buffer
	template < Device dev, bool markChanged = true >
	void reserve_index_buffer(std::size_t capacity);
	// Synchronizes two device index buffers
	template < Device changed, Device sync >
	void synchronize_index_buffer();
	template < Device dev >
	void unload_index_buffer();
	// Resizes the attribute buffer to hold v vertex and f face attribute pointers
	template < Device dev >
	void resizeAttribBuffer(std::size_t v, std::size_t f);

	VertexAttributePoolType m_vertexAttributes;
	FaceAttributePoolType m_faceAttributes;
	VertexAttributeHandle m_pointsHdl;
	VertexAttributeHandle m_normalsHdl;
	VertexAttributeHandle m_uvsHdl;
	std::atomic_uint32_t m_curvRefCount;
	std::optional<VertexAttributeHandle> m_curvatureHdl;
	std::optional<VertexAttributeHandle> m_animationWeightHdl;
	FaceAttributeHandle m_matIndicesHdl;
	// Vertex-index buffer, first for the triangles, then for quads
	IndexBuffers m_indexBuffer;
	// Array for aquired attribute descriptors
	AttribBuffers m_attribBuffer;

	// Dirty flags for descriptor rebuilding
	util::TaggedTuple<DescFlags<Device::CPU>, DescFlags<Device::CUDA>> m_descFlags;

	ei::Box m_boundingBox;
	std::size_t m_triangles;
	std::size_t m_quads;

	// Keeps track of whether displacement mapping was already applied or not
	bool m_wasDisplaced;
};

}}} // namespace mufflon::scene::geometry
