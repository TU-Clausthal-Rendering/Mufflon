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
		OmAttrHandle omHandle;
		AttributeHandle<Type> customHandle;
	};

	// Struct containing handles to both OpenMesh and custom attributes (faces)
	template < class T >
	struct FaceAttributeHandle {
		using Type = T;
		using OmAttrHandle = OpenMesh::FPropHandleT<Type>;
		OmAttrHandle omHandle;
		AttributeHandle<Type> customHandle;
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
	Polygons() :
		m_meshData(),
		m_pointsAttr(this->create_points_attribute()),
		m_normalsAttr(this->create_normals_attribute()),
		m_uvsAttr(this->create_uvs_attribute()),
		m_matIndexAttr(this->create_mat_index_attribute())
	{}
	// Creates polygon from already-created mesh.
	Polygons(MeshType&& mesh) :
		m_meshData(mesh),
		m_pointsAttr(this->create_points_attribute()),
		m_normalsAttr(this->create_normals_attribute()),
		m_uvsAttr(this->create_uvs_attribute()),
		m_matIndexAttr(this->create_mat_index_attribute())
	{}

	Polygons(const Polygons&) = default;
	Polygons(Polygons&&) = default;
	Polygons& operator=(const Polygons&) = delete;
	Polygons& operator=(Polygons&&) = delete;
	~Polygons() = default;

	void resize(std::size_t vertices, std::size_t edges, std::size_t faces) {
		// TODO
		m_meshData.resize(vertices, edges, faces);
	}

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
			return { attrHandle, m_attributes.add<Type>(name, omAttr) };
		} else {
			// Found in OpenMesh already, now find it in our list
			auto opt = m_attributes.find<Type>(name);
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

	// Adds a new vertex.
	VertexHandle add(const Point& point, const Normal& normal, const UvCoordinate& uv);
	// Adds a new triangle.
	TriangleHandle add(const VertexHandle &vh, const Triangle& tri, MaterialIndex idx);
	// Adds a new quad.
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
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class T >
	std::size_t add_bulk(const VertexAttributeHandle<T>& attrHandle,
						 const VertexHandle& startVertex, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.omHandle.is_valid());
		Attribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startVertex, count, attrStream);
	}
	// Also performs bulk-load for an attribute, but aquires it first.
	template < class T >
	std::size_t add_bulk(const FaceAttributeHandle<T>& attrHandle,
						 const FaceHandle& startFace, std::size_t count,
						 std::istream& attrStream) {
		mAssert(attrHandle.is_valid());
		Attribute<T>& attribute = this->aquire(attrHandle);
		return add_bulk(attribute, startFace, count, attrStream);
	}

	// Synchronizes the default attributes position, normal, uv, matindex 
	template < Device dev >
	void synchronize() {
		m_attributes.synchronize<dev>();
	}

	template < Device dev >
	void unload() {
		m_attributes.unload<dev>();
	}

	// TODO: unload

	// Implements tessellation for uniform subdivision.
	void tessellate(OpenMesh::Subdivider::Uniform::SubdividerT<MeshType, Real>& tessellater,
					std::size_t divisions);
	// Implements tessellation for adaptive subdivision.
	void tessellate(OpenMesh::Subdivider::Adaptive::CompositeT<MeshType>& tessellater,
					std::size_t divisions);
	// Implements decimation.
	void create_lod(OpenMesh::Decimater::DecimaterT<MeshType>& decimater,
					std::size_t target_vertices);

	// Gets a constant handle to the underlying mesh data.
	const MeshType& native() const {
		return m_meshData;
	}

private:
	// These methods simply create references to the attributes
	// TODO: by holding references to them, if they ever get removed, we're in a bad pot
	Attribute<OpenMesh::Vec3f>& create_points_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> pointsHdl = m_meshData.points_pph();
		mAssert(pointsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& pointsProp = m_meshData.property(pointsHdl);
		return m_attributes.aquire(m_attributes.add<OpenMesh::Vec3f>(pointsProp.name(), pointsProp));
	}

	Attribute<OpenMesh::Vec3f>& create_normals_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec3f> normalsHdl = m_meshData.vertex_normals_pph();
		mAssert(normalsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec3f>& normalsProp = m_meshData.property(normalsHdl);
		return m_attributes.aquire(m_attributes.add<OpenMesh::Vec3f>(normalsProp.name(), normalsProp));
	}

	Attribute<OpenMesh::Vec2f>& create_uvs_attribute() {
		OpenMesh::VPropHandleT<OpenMesh::Vec2f> uvsHdl = m_meshData.vertex_texcoords2D_pph();
		mAssert(uvsHdl.is_valid());
		OpenMesh::PropertyT<OpenMesh::Vec2f>& uvsProp = m_meshData.property(uvsHdl);
		return m_attributes.aquire(m_attributes.add<OpenMesh::Vec2f>(uvsProp.name(), uvsProp));
	}

	Attribute<MaterialIndex>& create_mat_index_attribute() {
		auto handle = this->request<FaceAttributeHandle<MaterialIndex>>("materialIndex");
		return m_attributes.aquire(handle.customHandle);
	}

	MeshType m_meshData;
	AttributeListType m_attributes;
	Attribute<OpenMesh::Vec3f>& m_pointsAttr;
	Attribute<OpenMesh::Vec3f>& m_normalsAttr;
	Attribute<OpenMesh::Vec2f>& m_uvsAttr;
	Attribute<MaterialIndex>& m_matIndexAttr;
};

} // namespace mufflon::scene::geometry