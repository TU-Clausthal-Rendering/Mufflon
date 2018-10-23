#pragma once

#include "ei/vector.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <optional>
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

namespace mufflon::scene::geometry {


/// Traits for a polygon mesh - has to have normals and 2D UV coordinates per vertex.
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
	using Point = ei::Vec3;
	using Normal = ei::Vec3;
	using UvCoordinate = ei::Vec2;
	using Index = u32;
	using MaterialIndex = u32;
	using Triangle = std::array<Index, 3u>;
	using Quad = std::array<Index, 4u>;
	// OpenMesh types
	template < class Type >
	using Attribute = OpenMesh::PropertyT<Type>;
	template < class Type >
	using VertexAttributeHandle = OpenMesh::VPropHandleT<Type>;
	template < class Type >
	using FaceAttributeHandle = OpenMesh::FPropHandleT<Type>;
	using VertexHandle = OpenMesh::VertexHandle;
	using FaceHandle = OpenMesh::FaceHandle;
	using TriangleHandle = OpenMesh::FaceHandle;
	using QuadHandle = OpenMesh::FaceHandle;
	using MeshType = OpenMesh::PolyMesh_ArrayKernelT<PolygonTraits>;
	using VertexBulkReturn = std::tuple<VertexHandle, std::size_t,
		std::size_t, std::size_t>;

	// Ensure matching data types
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Point)
				  && sizeof(Point) == 3u * sizeof(float)
				  && alignof(OpenMesh::Vec3f) == alignof(Point),
				  "Point type must be compatible to OpenMesh");
	static_assert(sizeof(OpenMesh::Vec3f) == sizeof(Normal)
				  && sizeof(Normal) == 3u * sizeof(float)
				  && alignof(OpenMesh::Vec3f) == alignof(Normal),
				  "Normal type must be compatible to OpenMesh");

	/// Default construction, creates material-index attribute.
	Polygons() :
		m_meshData(),
		m_matIndices(m_meshData.property(this->create_matidx_attrib()))
	{}
	/// Creates polygon from already-created mesh.
	Polygons(MeshType&& mesh) :
		m_meshData(mesh),
		m_matIndices(m_meshData.property(this->create_matidx_attrib()))
	{}
	Polygons(const Polygons&) = default;
	Polygons(Polygons&&) = default;
	Polygons& operator=(const Polygons&) = default;
	Polygons& operator=(Polygons&&) = default;
	~Polygons() = default;
	
	void reserve(std::size_t vertices, std::size_t edges, std::size_t faces) {
		m_meshData.reserve(vertices, edges, faces);
	}

	void resize(std::size_t vertices, std::size_t edges, std::size_t faces) {
		m_meshData.resize(vertices, edges, faces);
	}

	void clear() {
		m_meshData.clear();
	}

	/// Requests a new attribute, either for face or vertex.
	template < class AttributeHandle >
	AttributeHandle request(const std::string& name) {
		AttributeHandle attr_handle;
		if(!m_meshData.get_property_handle(attr_handle, name))
			m_meshData.add_property(attr_handle, name);
		return attr_handle;
	}

	template < class AttributeHandle >
	void remove(const AttributeHandle &attr) {
		m_meshData.remove_property(attr);
	}

	template < class AttributeHandle >
	std::optional<AttributeHandle> find(const std::string& name) {
		AttributeHandle attr;
		if(!m_meshData.get_property_handle(attr, name))
			return std::nullopt;
		return attr;
	}

	template < template < class > class AttributeHandle, class Type >
	Attribute<Type> &aquire(const AttributeHandle<Type>& attrHandle) {
		return m_meshData.property(attrHandle);
	}

	template < template < class > class AttributeHandle, class Type >
	const Attribute<Type> &aquire(const AttributeHandle<Type>& attrHandle) const {
		return m_meshData.property(attrHandle);
	}

	/// Adds a new vertex.
	VertexHandle add(const Point& point, const Normal& normal, const UvCoordinate& uv);
	/// Adds a new triangle.
	TriangleHandle add(const VertexHandle &vh, const Triangle& tri, MaterialIndex idx);
	/// Adds a new quad.
	QuadHandle add(const VertexHandle &vh, const Quad& quad, MaterialIndex idx);

	/**
	 * Adds a bulk of vertices.
	 * Returns both a handle to the first added vertex as well as the number of
	 * read vertices.
	 */
	VertexBulkReturn add_bulk(std::size_t count, std::istream& pointStream,
							  std::istream& normalStream, std::istream& uvStream);
	/**
	 * Bulk-loads the given attribute starting at the given vertex.
	 * The number of read values will be capped by the number of vertice present
	 * after the starting position.
	 */
	template < class Type >
	std::size_t add_bulk(const Attribute<Type>& attribute, const VertexHandle& startVertex,
						 std::size_t count, std::istream& attrStream) {
		mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData.n_vertices());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData.n_vertices() - static_cast<std::size_t>(startVertex.idx()),
										   count);
		// Read the attribute from the stream
		attrStream.read(attribute.data_vector().data(), actualCount * sizeof(Type));
		std::size_t actuallyRead = static_cast<std::size_t>(attrStream.gcount()) / sizeof(Type);
		return actuallyRead;
	}
	/**
	 * Bulk-loads the given attribute starting at the given face.
	 * The number of read values will be capped by the number of faces present
	 * after the starting position.
	 */
	template < class Type >
	std::size_t add_bulk(const Attribute<Type>& attribute, const FaceHandle& startFace,
						 std::size_t count, std::istream& attrStream) {
		mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData.n_faces());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData.n_faces() - static_cast<std::size_t>(startFace.idx()),
										   count);
		// Read the attribute from the stream
		attrStream.read(attribute.data_vector().data(), actualCount * sizeof(Type));
		std::size_t actuallyRead = static_cast<std::size_t>(attrStream.gcount()) / sizeof(Type);
		return actuallyRead;
	}
	/// Also performs bulk-load for an attribute, but aquires it first.
	template < class Type >
	std::size_t add_bulk(const VertexAttributeHandle<Type>& attrHandle,
						 const VertexHandle& startVertex, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.is_valid());
		Attribute<Type>& attribute = m_meshData.property(attrHandle);
		return add_bulk(attribute, startVertex, count, attrStream);
	}
	/// Also performs bulk-load for an attribute, but aquires it first.
	template < class Type >
	std::size_t add_bulk(const FaceAttributeHandle<Type>& attrHandle,
						 const FaceHandle& startFace, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.is_valid());
		Attribute<Type>& attribute = m_meshData.property(attrHandle);
		return add_bulk(attribute, startFace, count, attrStream);
	}

	template < template < class > class AttributeHandle, class Type >
	const Type& get(const VertexHandle& vertexHandle, const AttributeHandle<Type>& attrHandle) const {
		m_meshData.property(attrHandle);
		return m_meshData.property(attrHandle, vertexHandle);
	}

	template < template < class > class AttributeHandle, class Type >
	void set(const VertexHandle& vertexHandle, const AttributeHandle<Type>& attrHandle,
			 const Type& val) {
		m_meshData.property(attrHandle, vertexHandle) = val;
	}

	/// Implements tessellation for uniform subdivision.
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<MeshType, Real>& tessellater,
					std::size_t divisions);
	/// Implements tessellation for adaptive subdivision.
	void tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
					std::size_t divisions);
	/// Implements decimation.
	void create_lod(OpenMesh::Decimater::DecimaterT<MeshType>& decimater,
					std::size_t target_vertices);

	/// Gets a constant handle to the underlying mesh data.
	const MeshType& native() const {
		return m_meshData;
	}

private:
	/// Creates the material index attribute
	FaceAttributeHandle<MaterialIndex> create_matidx_attrib() {
		FaceAttributeHandle<MaterialIndex> attrHandle;
		m_meshData.add_property(attrHandle, "materialIdx");
		if(!attrHandle.is_valid())
			throw std::runtime_error("Failed to add material index attribute!");
		return attrHandle;
	}

	MeshType m_meshData;
	Attribute<MaterialIndex>& m_matIndices;
};

} // namespace mufflon::scene::geometry