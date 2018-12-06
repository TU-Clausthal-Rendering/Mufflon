#pragma once

#include "polygon_mesh.hpp"
#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "core/scene/attribute_list.hpp"
#include "util/range.hpp"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <tuple>
#include <optional>

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
	using VertexAttributeListType = OmAttributeList<false>;
	using FaceAttributeListType = OmAttributeList<true>;
	template < class Attr >
	using VertexAttributeHdl = typename VertexAttributeListType::template AttributeHandle<Attr>;
	template < class Attr >
	using FaceAttributeHdl = typename FaceAttributeListType::template AttributeHandle<Attr>;
	template < class T >
	using VertexAttribute = typename VertexAttributeListType::template BaseAttribute<T>;
	template < class T >
	using FaceAttribute = typename FaceAttributeListType::template BaseAttribute<T>;
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

	// Struct containing handles to both OpenMesh and custom attributes (vertex)
	template < class T >
	struct VertexAttributeHandle {
		using Type = T;
		using OmAttrHandle = OpenMesh::VPropHandleT<Type>;
		using CustomAttrHandle = VertexAttributeHdl<Type>;

		OmAttrHandle omHandle;
		CustomAttrHandle customHandle;
	};

	// Struct containing handles to both OpenMesh and custom attributes (faces)
	template < class T >
	struct FaceAttributeHandle {
		using Type = T;
		using OmAttrHandle = OpenMesh::FPropHandleT<Type>;
		using CustomAttrHandle = FaceAttributeHdl<Type>;
		OmAttrHandle omHandle;
		CustomAttrHandle customHandle;
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
	Polygons(Polygons&&) = default;
	Polygons& operator=(const Polygons&) = delete;
	Polygons& operator=(Polygons&&) = delete;
	~Polygons() = default;

	void resize(std::size_t vertices, std::size_t edges, std::size_t faces);

	// Requests a new attribute, either for face or vertex.
	template < class AttrHandle >
	AttrHandle request(const std::string& name) {
		using Type = typename AttrHandle::Type;

		typename AttrHandle::OmAttrHandle attrHandle;
		if(!m_meshData->get_property_handle(attrHandle, name)) {
			// Add the attribute to OpenMesh...
			m_meshData->add_property(attrHandle, name);
			// ...as well as our attribute list
			return { attrHandle, select_list<AttrHandle>().template add<Type>(name, attrHandle) };
		} else {
			// Found in OpenMesh already, now find it in our list
			auto opt = select_list<AttrHandle>().template find<Type>(name);
			mAssertMsg(opt.has_value(), "This should never ever happen that we have a "
					   "property in OpenMesh but not in our custom list");
			return { attrHandle, opt.value() };
		}
	}

	template < class AttributeHandle >
	void remove(AttributeHandle& attr) {
		// Remove from both lists
		m_meshData->remove_property(attr.omHandle);
		select_list<AttributeHandle>().remove(attr.customHandle);
	}

	template < class AttrHandle >
	std::optional<AttrHandle> find(const std::string& name) {
		using Type = typename AttrHandle::Type;

		typename AttrHandle::OmAttrHandle attrHandle;
		if(!m_meshData->get_property_handle(attrHandle, name))
			return std::nullopt;
		// Find attribute in custom list as 
		auto opt = select_list<AttrHandle>().template find<Type>(name);
		mAssert(opt.has_value());
		return AttrHandle{
			attrHandle,
			opt.value()
		};
	}

	template < class T >
	auto& aquire(const VertexAttributeHandle<T>& attrHandle) {
		return m_vertexAttributes.aquire(attrHandle.customHandle);
	}

	template < class T >
	const auto& aquire(const VertexAttributeHandle<T>& attrHandle) const {
		return m_vertexAttributes.aquire(attrHandle.customHandle);
	}

	template < class T >
	auto& aquire(const FaceAttributeHandle<T>& attrHandle) {
		return m_faceAttributes.aquire(attrHandle.customHandle);
	}

	template < class T >
	const auto& aquire(const FaceAttributeHandle<T>& attrHandle) const {
		return m_faceAttributes.aquire(attrHandle.customHandle);
	}

	/**
	 * Returns a descriptor (on CPU side) with pointers to resources (on Device side).
	 * Takes two tuples: they must each contain the aquired attributes which the
	 * renderer wants to have access to.
	 */
	template < Device dev, class... VArgs, class... FArgs >
	PolymeshDescriptor<dev> get_descriptor(std::tuple<VArgs...>& vertexAttribs,
									  std::tuple<FArgs...>& faceAttribs) {
		this->synchronize<dev>();
		constexpr std::size_t numVertexAttribs = sizeof...(VArgs);
		constexpr std::size_t numFaceAttribs = sizeof...(FArgs);
		void** devVertexAttribs = nullptr;
		void** devFaceAttribs = nullptr;
		// Collect the attributes; for that, we iterate the given Attributes and
		// gather them on CPU side (or rather, their device pointers); then
		// we copy it to the actual device
		if(numVertexAttribs > 0) {
			std::vector<void*> cpuVertAttibs(numVertexAttribs);
			push_back_attrib<0u, dev>(cpuVertAttibs, vertexAttribs);
			devVertexAttribs = Allocator<dev>::template alloc_array<void*>(numVertexAttribs);
			Allocator<Device::CPU>::template copy<void*, dev>(devVertexAttribs, cpuVertAttibs.data(),
													 numVertexAttribs);
		}
		if(numFaceAttribs > 0) {
			std::vector<void*> cpuFaceAttibs(numFaceAttribs);
			push_back_attrib<0u, dev>(cpuFaceAttibs, faceAttribs);
			devFaceAttribs = Allocator<dev>::template alloc_array<void*>(numFaceAttribs);
			Allocator<Device::CPU>::template copy<void*, dev>(devFaceAttribs, cpuFaceAttibs.data(),
													 numFaceAttribs);
		}
		
		// TODO: face indices
		return PolymeshDescriptor{
			static_cast<u32>(this->get_vertex_count()),
			static_cast<u32>(this->get_triangle_count()),
			static_cast<u32>(this->get_quad_count()),
			static_cast<u32>(numVertexAttribs),
			static_cast<u32>(numFaceAttribs),
			reinterpret_cast<const ei::Vec3*>(*this->get_points().aquireConst<dev>()),
			reinterpret_cast<const ei::Vec3*>(*this->get_normals().aquireConst<dev>()),
			reinterpret_cast<const ei::Vec2*>(*this->get_uvs().aquireConst<dev>()),
			*this->get_mat_indices().aquireConst<dev>(),
			this->get_index_buffer(),
			devVertexAttribs,
			devFaceAttribs
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
	 * Bulk-loads the given attribute starting at the given vertex.
	 * The number of read values will be capped by the number of vertice present
	 * after the starting position.
	 */
	template < class Attribute >
	std::size_t add_bulk(Attribute& attribute, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream) {
		mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData->n_vertices());
		// Cap the number of attributes
		const std::size_t actualCount = std::min(m_meshData->n_vertices() - static_cast<std::size_t>(startVertex.idx()),
												 count);
		// Read the attribute from the stream
		return attribute.restore(attrStream,
								 static_cast<std::size_t>(startVertex.idx()), actualCount);
	}
	/**
	 * Bulk-loads the given attribute starting at the given face.
	 * The number of read values will be capped by the number of faces present
	 * after the starting position.
	 */
	template < class Attribute >
	std::size_t add_bulk(Attribute& attribute, const FaceHandle& startFace,
						 std::size_t count, util::IByteReader& attrStream) {
		mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData->n_faces());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData->n_faces() - static_cast<std::size_t>(startFace.idx()),
										   count);
		// Read the attribute from the stream
		return attribute.restore(attrStream,
								 static_cast<std::size_t>(startFace.idx()), actualCount);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class T >
	std::size_t add_bulk(const VertexAttributeHandle<T>& attrHandle,
						 const VertexHandle& startVertex, std::size_t count,
						 util::IByteReader& attrStream) {
		mAssert(attrHandle.omHandle.is_valid());
		VertexAttribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startVertex, count, attrStream);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class T >
	std::size_t add_bulk(const FaceAttributeHandle<T>& attrHandle,
						 const FaceHandle& startFace, std::size_t count,
						 util::IByteReader& attrStream) {
		mAssert(attrHandle.omHandle.is_valid());
		FaceAttribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startFace, count, attrStream);
	}

	VertexAttribute<OpenMesh::Vec3f>& get_points() {
		return this->aquire(m_pointsAttrHdl);
	}
	const VertexAttribute<OpenMesh::Vec3f>& get_points() const {
		return this->aquire(m_pointsAttrHdl);
	}

	VertexAttribute<OpenMesh::Vec3f>& get_normals() {
		return this->aquire(m_normalsAttrHdl);
	}
	const VertexAttribute<OpenMesh::Vec3f>& get_normals() const {
		return this->aquire(m_normalsAttrHdl);
	}

	VertexAttribute<OpenMesh::Vec2f>& get_uvs() {
		return this->aquire(m_uvsAttrHdl);
	}
	const VertexAttribute<OpenMesh::Vec2f>& get_uvs() const {
		return this->aquire(m_uvsAttrHdl);
	}

	FaceAttribute<MaterialIndex>& get_mat_indices() {
		return this->aquire(m_matIndexAttrHdl);
	}
	const FaceAttribute<MaterialIndex>& get_mat_indices() const {
		return this->aquire(m_matIndexAttrHdl);
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_vertexAttributes.synchronize<dev>();
	}

	template < Device dev >
	void unload() {
		m_vertexAttributes.unload<dev>();
	}

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
		return m_vertexAttributes.get_num_attributes();
	}

	std::size_t get_face_attribute_count() const noexcept {
		return m_faceAttributes.get_num_attributes();
	}

private:
	// Helper struct for identifying handle type
	template < class >
	struct IsvertexHandleType : std::false_type {};
	template < class T >
	struct IsvertexHandleType<VertexAttributeHandle<T>> : std::true_type {};

	// Helper function for adding attributes since functions cannot be partially specialized
	template < class AttributeHandle >
	auto& select_list() {
		if constexpr(IsvertexHandleType<AttributeHandle>::value)
			return m_vertexAttributes;
		else
			return m_faceAttributes;
	}

	// Helper for iterating a tuple
	template < std::size_t I, Device dev, class... Args >
	static void push_back_attrib(std::vector<void*>& vec, std::tuple<Args...>& attribs) {
		if constexpr(I < sizeof...(Args)) {
			vec.push_back(*std::get<I>(attribs).aquire<dev>());
			push_back_attrib<I + 1u, dev>(vec, attribs);
		}
	}

	// Helper class for distinct array handle types
	template < Device dev >
	struct IndexBuffer {
		ArrayDevHandle_t<dev, u32> indices;
	};

	// These methods simply create references to the attributes
	// By holding references to them, if they ever get removed, we're in a bad spot
	// So you BETTER not remove the standard attributes
	VertexAttributeHandle<OpenMesh::Vec3f> create_points_handle();
	VertexAttributeHandle<OpenMesh::Vec3f> create_normals_handle();
	VertexAttributeHandle<OpenMesh::Vec2f> create_uvs_handle();
	FaceAttributeHandle<MaterialIndex> create_mat_index_handle();

	// It's a unique pointer so we have one fixed address we can reference in OmAttributePool
	// TODO: does that degrade performance? probably not, since attributes aren't aquired often
	std::unique_ptr<PolygonMeshType> m_meshData;
	VertexAttributeListType m_vertexAttributes;
	FaceAttributeListType m_faceAttributes;
	VertexAttributeHandle<OpenMesh::Vec3f> m_pointsAttrHdl;
	VertexAttributeHandle<OpenMesh::Vec3f> m_normalsAttrHdl;
	VertexAttributeHandle<OpenMesh::Vec2f> m_uvsAttrHdl;
	FaceAttributeHandle<MaterialIndex> m_matIndexAttrHdl;
	// Vertex-index buffer, first for the triangles, then for quads
	util::TaggedTuple<IndexBuffer<Device::CPU>, IndexBuffer<Device::CUDA>> m_indexBuffer;
	ei::Box m_boundingBox;
	std::size_t m_triangles = 0u;
	std::size_t m_quads = 0u;
};

} // namespace mufflon::scene::geometry