#pragma once

#include "ei/3dtypes.hpp"
#include "util/assert.hpp"
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


// Traits for a polygon mesh - has to have normals and 2D UV coordinates per vertex.
struct PolygonTraits : public OpenMesh::DefaultTraits {
	VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::TexCoord2D);
};

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
	using AttributeListType = AttributeList<true>;
	template < class Attr >
	using AttributeHandle = typename AttributeListType::template AttributeHandle<Attr>;
	template < class T >
	using Attribute = typename AttributeListType::template Attribute<T>;
	using VertexHandle = OpenMesh::VertexHandle;
	using FaceHandle = OpenMesh::FaceHandle;
	using TriangleHandle = OpenMesh::FaceHandle;
	using QuadHandle = OpenMesh::FaceHandle;
	using MeshType = OpenMesh::PolyMesh_ArrayKernelT<PolygonTraits>;

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
		using CustomAttrHandle = AttributeHandle<Type>;

		OmAttrHandle omHandle;
		CustomAttrHandle customHandle;
	};

	// Struct containing handles to both OpenMesh and custom attributes (faces)
	template < class T >
	struct FaceAttributeHandle {
		using Type = T;
		using OmAttrHandle = OpenMesh::FPropHandleT<Type>;
		using CustomAttrHandle = AttributeHandle<Type>;
		OmAttrHandle omHandle;
		CustomAttrHandle customHandle;
	};

	class FaceIterator {
	public:
		static FaceIterator cbegin(const MeshType& mesh) {
			return FaceIterator(mesh, mesh.faces().begin());
		}

		static FaceIterator cend(const MeshType& mesh) {
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
		FaceIterator(const MeshType& mesh, OpenMesh::PolyConnectivity::ConstFaceIter iter) :
			m_mesh(mesh),
			m_faceIter(std::move(iter))
		{}

		const MeshType& m_mesh;
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
	Polygons(MeshType&& mesh);

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
		if(!m_meshData.get_property_handle(attrHandle, name)) {
			// Add the attribute to OpenMesh...
			m_meshData.add_property(attrHandle, name);
			// ...as well as our attribute list
			OpenMesh::PropertyT<Type> &omAttr = m_meshData.property(attrHandle);
			return { attrHandle, select_list<AttrHandle>().add(name, omAttr) };
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
		m_meshData.remove_property(attr.omHandle);
		select_list<AttributeHandle>().remove(attr.customHandle);
	}

	template < class AttrHandle >
	std::optional<AttrHandle> find(const std::string& name) {
		using Type = typename AttrHandle::Type;

		typename AttrHandle::OmAttrHandle attrHandle;
		if(!m_meshData.get_property_handle(attrHandle, name))
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
	/**
	 * Bulk-loads the given attribute starting at the given vertex.
	 * The number of read values will be capped by the number of vertice present
	 * after the starting position.
	 */
	template < class Type >
	std::size_t add_bulk(Attribute<Type>& attribute, const VertexHandle& startVertex,
						 std::size_t count, util::IByteReader& attrStream) {
		mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData.n_vertices());
		// Cap the number of attributes
		const std::size_t actualCount = std::min(m_meshData.n_vertices() - static_cast<std::size_t>(startVertex.idx()),
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
	template < class Type >
	std::size_t add_bulk(Attribute<Type>& attribute, const FaceHandle& startFace,
						 std::size_t count, util::IByteReader& attrStream) {
		mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData.n_faces());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData.n_faces() - static_cast<std::size_t>(startFace.idx()),
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
		Attribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startVertex, count, attrStream);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class T >
	std::size_t add_bulk(const FaceAttributeHandle<T>& attrHandle,
						 const FaceHandle& startFace, std::size_t count,
						 util::IByteReader& attrStream) {
		mAssert(attrHandle.omHandle.is_valid());
		Attribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startFace, count, attrStream);
	}

	Attribute<OpenMesh::Vec3f>& get_points() {
		return m_pointsAttr;
	}
	const Attribute<OpenMesh::Vec3f>& get_points() const {
		return m_pointsAttr;
	}

	Attribute<OpenMesh::Vec3f>& get_normals() {
		return m_normalsAttr;
	}
	const Attribute<OpenMesh::Vec3f>& get_normals() const {
		return m_normalsAttr;
	}

	Attribute<OpenMesh::Vec2f>& get_uvs() {
		return m_uvsAttr;
	}
	const Attribute<OpenMesh::Vec2f>& get_uvs() const {
		return m_uvsAttr;
	}

	Attribute<MaterialIndex>& get_mat_indices() {
		return m_matIndexAttr;
	}
	const Attribute<MaterialIndex>& get_mat_indices() const {
		return m_matIndexAttr;
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
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<MeshType, Real>& tessellater,
					std::size_t divisions);
	// Implements tessellation for adaptive subdivision.
	/*void tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
					std::size_t divisions);*/
	// Implements decimation.
	void create_lod(OpenMesh::Decimater::DecimaterT<MeshType>& decimater,
					std::size_t target_vertices);

	// Gets a constant handle to the underlying mesh data.
	const MeshType& native() const {
		return m_meshData;
	}

	// Get iterator over all faces (and vertices for the faces)
	util::Range<FaceIterator> faces() const {
		return util::Range<FaceIterator>{
			FaceIterator::cbegin(m_meshData),
			FaceIterator::cend(m_meshData)
		};
	}

	const ei::Box& get_bounding_box() const noexcept {
		return m_boundingBox;
	}

	std::size_t get_vertex_count() const noexcept {
		return m_meshData.n_vertices();
	}

	std::size_t get_edge_count() const noexcept {
		return m_meshData.n_edges();
	}

	std::size_t get_triangle_count() const noexcept {
		return m_triangles;
	}

	std::size_t get_quad_count() const noexcept {
		return m_quads;
	}

	std::size_t get_face_count() const noexcept {
		return m_meshData.n_faces();
	}

private:
	// Helper struct for identifying handle type
	template < class >
	struct IsvertexHandleType : std::false_type {};
	template < class T >
	struct IsvertexHandleType<VertexAttributeHandle<T>> : std::true_type {};

	// Helper function for adding attributes since functions cannot be partially specialized
	template < class AttributeHandle >
	AttributeListType& select_list() {
		if constexpr(IsvertexHandleType<AttributeHandle>::value)
			return m_vertexAttributes;
		else
			return m_faceAttributes;
	}

	// These methods simply create references to the attributes
	// By holding references to them, if they ever get removed, we're in a bad spot
	// So you BETTER not remove the standard attributes
	VertexAttributeHandle<OpenMesh::Vec3f> create_points_handle() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> omHandle = m_meshData.points_pph();
		mAssert(omHandle.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& pointsProp = m_meshData.property(omHandle);
		AttributeHandle<OpenMesh::Vec3f> customHandle = m_vertexAttributes.add<OpenMesh::Vec3f>(pointsProp.name(), pointsProp);
		return { std::move(omHandle), std::move(customHandle) };
	}

	VertexAttributeHandle<OpenMesh::Vec3f> create_normals_handle() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> omHandle = m_meshData.vertex_normals_pph();
		mAssert(omHandle.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& normalsProp = m_meshData.property(omHandle);
		AttributeHandle<OpenMesh::Vec3f> customHandle = m_vertexAttributes.add<OpenMesh::Vec3f>(normalsProp.name(), normalsProp);
		return { std::move(omHandle), std::move(customHandle) };
	}

	VertexAttributeHandle<OpenMesh::Vec2f> create_uvs_handle() {
		OpenMesh::VPropHandleT<OpenMesh::Vec2f> omHandle = m_meshData.vertex_texcoords2D_pph();
		mAssert(omHandle.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec2f>& uvsProp = m_meshData.property(omHandle);
		AttributeHandle<OpenMesh::Vec2f> customHandle = m_vertexAttributes.add<OpenMesh::Vec2f>(uvsProp.name(), uvsProp);
		return { std::move(omHandle), std::move(customHandle) };
	}

	FaceAttributeHandle<MaterialIndex> create_mat_index_handle() {
		return this->request<FaceAttributeHandle<MaterialIndex>>("materialIndex");
	}

	MeshType m_meshData;
	AttributeListType m_vertexAttributes;
	AttributeListType m_faceAttributes;
	Attribute<OpenMesh::Vec3f>& m_pointsAttr;
	Attribute<OpenMesh::Vec3f>& m_normalsAttr;
	Attribute<OpenMesh::Vec2f>& m_uvsAttr;
	Attribute<MaterialIndex>& m_matIndexAttr;
	ei::Box m_boundingBox;
	std::size_t m_triangles;
	std::size_t m_quads;
};

} // namespace mufflon::scene::geometry