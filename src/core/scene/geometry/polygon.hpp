#pragma once

#include "ei/vector.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"
#include "core/scene/types.hpp"
#include "core/scene/attribute_list.hpp"
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
	using Index = u32;
	using Triangle = std::array<Index, 3u>;
	using Quad = std::array<Index, 4u>;
	// OpenMesh types
	using AttributeListType = AttributeList<false>;
	using VertexHandle = OpenMesh::VertexHandle;
	using FaceHandle = OpenMesh::FaceHandle;
	using TriangleHandle = OpenMesh::FaceHandle;
	using QuadHandle = OpenMesh::FaceHandle;
	using MeshType = OpenMesh::PolyMesh_ArrayKernelT<PolygonTraits>;
	using VertexBulkReturn = std::tuple<VertexHandle, std::size_t,
		std::size_t, std::size_t>;

	// Struct containing handles to both OpenMesh and custom attributes (vertex)
	template < class T, template < class, bool > class Attr >
	struct VertexAttributeHandle {
		using Type = T;
		template < class U, bool b >
		using Attribute = Attr<U, b>;
		using OmAttrHandle = OpenMesh::VPropHandleT<T>;
		OmAttrHandle omHandle;
		AttributeListType::AttributeHandle<Attr<T, false>> customHandle;
	};

	// Struct containing handles to both OpenMesh and custom attributes (faces)
	template < class T, template < class, bool > class Attr >
	struct FaceAttributeHandle {
		using Type = T;
		template < class U, bool b >
		using Attribute = Attr<U, b>;
		using OmAttrHandle = OpenMesh::FPropHandleT<T>;
		OmAttrHandle omHandle;
		AttributeListType::AttributeHandle<Attr<T, false>> customHandle;
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

	/// Default construction, creates material-index attribute.
	Polygons() :
		m_meshData(),
		m_pointsAttr(this->create_points_attribute()),
		m_normalsAttr(this->create_normals_attribute()),
		m_uvsAttr(this->create_uvs_attribute()),
		m_matIndexAttr(this->create_mat_index_attribute())
	{}
	/// Creates polygon from already-created mesh.
	Polygons(MeshType&& mesh) :
		m_meshData(mesh),
		m_pointsAttr(this->create_points_attribute()),
		m_normalsAttr(this->create_normals_attribute()),
		m_uvsAttr(this->create_uvs_attribute()),
		m_matIndexAttr(this->create_mat_index_attribute())
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
	/*template < class AttributeHandle >
	AttributeHandle request(const std::string& name) {
		AttributeHandle attr_handle;
		if (!m_meshData.get_property_handle(attr_handle, name)) {
			m_meshData.add_property(attr_handle, name);
			// Since we track consistency in our custom attributes, we create one of them
			// and exchange the vectors. Caution: this relies on PropertyT to just be a
			// wrapper around a vector!
		}
		return attr_handle;
	}*/

	// Requests a new attribute, either for face or vertex.
	template < class AttributeHandle >
	AttributeHandle request(const std::string& name) {
		using Type = typename AttributeHandle::Type;

		typename AttributeHandle::OmAttrHandle attrHandle;
		if(!m_meshData.get_property_handle(attrHandle, name)) {
			// Add the attribute to OpenMesh...
			m_meshData.add_property(attrHandle, name);
			// ...as well as our attribute list
			OpenMesh::PropertyT<Type> &omAttr = m_meshData.property(attrHandle);
			return { attrHandle, m_attributes.add<typename AttributeHandle::template Attribute, Type>(name, omAttr.data_vector()) };
		} else {
			// Found in OpenMesh already, now find it in our list
			auto opt = m_attributes.find<typename AttributeHandle::template Attribute<Type, false>>(name);
			mAssertMsg(opt.has_value(), "This should never ever happen that we have a "
					   "property in OpenMesh but not in our custom list");
			return { attrHandle, opt.value() };
		}
	}

	template < class AttributeHandle >
	void remove(const AttributeHandle &attr) {
		// Remove from both lists
		m_meshData.remove_property(attr.omHandle);
		m_attributes.remove(attr.customHandle);
	}

	template < class AttributeHandle >
	std::optional<AttributeHandle> find(const std::string& name) {
		using Type = typename AttributeHandle::Type;
		using Attribute = typename AttributeHandle::Attribute;

		typename AttributeHandle::OmAttrHandle attrHandle;
		if(!m_meshData.get_property_handle(attrHandle, name))
			return std::nullopt;
		// Find attribute in custom list as 
		auto opt = m_attributes.find<Attribute<Type, false>>(name);
		mAssertMsg(opt.has_value(), "This should never ever happen that we have a "
				   "property in OpenMesh but not in our custom list");
		return { attrHandle, opt.value() };
	}

	template < class AttributeHandle >
	auto& aquire(const AttributeHandle& attrHandle) {
		return m_attributes.aquire(attrHandle.customHandle);
	}

	template < class AttributeHandle >
	const auto& aquire(const AttributeHandle& attrHandle) const {
		return m_attributes.aquire(attrHandle.customHandle);
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
	template < template < class > class Attribute, class Type >
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
	template < template < class > class Attribute, class Type >
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
	template < class T, template < class, bool > class Attr >
	std::size_t add_bulk(const VertexAttributeHandle<T, Attr>& attrHandle,
						 const VertexHandle& startVertex, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.omHandle.is_valid());
		Attr<T, false>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startVertex, count, attrStream);
	}
	/// Also performs bulk-load for an attribute, but aquires it first.
	template < class T, template < class, bool > class Attr >
	std::size_t add_bulk(const FaceAttributeHandle<T, Attr>& attrHandle,
						 const FaceHandle& startFace, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.is_valid());
		Attr<T, false>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startFace, count, attrStream);
	}

	// Implements tessellation for uniform subdivision.
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<MeshType, Real>& tessellater,
					std::size_t divisions);
	// Implements tessellation for adaptive subdivision.
	void tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
					std::size_t divisions);
	// Implements decimation.
	void create_lod(OpenMesh::Decimater::DecimaterT<MeshType>& decimater,
					std::size_t target_vertices);

	/// Gets a constant handle to the underlying mesh data.
	const MeshType& native() const {
		return m_meshData;
	}

private:
	ArrayAttribute<OpenMesh::Vec3f, false>& create_points_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> pointsHdl = m_meshData.points_pph();
		mAssert(pointsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& pointsProp = m_meshData.property(pointsHdl);
		return m_attributes.aquire(m_attributes.add<ArrayAttribute, OpenMesh::Vec3f>(pointsProp.name(), pointsProp.data_vector()));
	}

	ArrayAttribute<OpenMesh::Vec3f, false>& create_normals_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> normalsHdl = m_meshData.vertex_normals_pph();
		mAssert(normalsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& normalsProp = m_meshData.property(normalsHdl);
		return m_attributes.aquire(m_attributes.add<ArrayAttribute, OpenMesh::Vec3f>(normalsProp.name(), normalsProp.data_vector()));
	}

	ArrayAttribute<OpenMesh::Vec2f, false>& create_uvs_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec2f> uvsHdl = m_meshData.vertex_texcoords2D_pph();
		mAssert(uvsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec2f>& uvsProp = m_meshData.property(uvsHdl);
		return m_attributes.aquire(m_attributes.add<ArrayAttribute, OpenMesh::Vec2f>(uvsProp.name(), uvsProp.data_vector()));
	}

	ArrayAttribute<MaterialIndex, false>& create_mat_index_attribute() {
		auto handle = this->request<FaceAttributeHandle<MaterialIndex, ArrayAttribute>>("materialIndex");
		return m_attributes.aquire(handle.customHandle);
	}

	MeshType m_meshData;
	AttributeListType m_attributes;
	ArrayAttribute<OpenMesh::Vec3f, false>& m_pointsAttr;
	ArrayAttribute<OpenMesh::Vec3f, false>& m_normalsAttr;
	ArrayAttribute<OpenMesh::Vec2f, false>& m_uvsAttr;
	ArrayAttribute<MaterialIndex, false>& m_matIndexAttr;
};

} // namespace mufflon::scene::geometry