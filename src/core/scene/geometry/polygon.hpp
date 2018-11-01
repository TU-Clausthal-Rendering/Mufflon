#pragma once

#include "export/dll_export.hpp"
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
class LIBRARY_API Polygons {
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
		m_pointsHdl(this->create_points_handle()),
		m_normalsHdl(this->create_normals_handle()),
		m_uvsHdl(this->create_uvs_handle()),
		m_matIndexHdl(this->create_mat_index_handle()),
		m_pointsAttr(this->aquire(m_pointsHdl)),
		m_normalsAttr(this->aquire(m_normalsHdl)),
		m_uvsAttr(this->aquire(m_uvsHdl)),
		m_matIndexAttr(this->aquire(m_matIndexHdl))
	{}
	// Creates polygon from already-created mesh.
	Polygons(MeshType&& mesh) :
		m_meshData(mesh),
		m_pointsHdl(this->create_points_handle()),
		m_normalsHdl(this->create_normals_handle()),
		m_uvsHdl(this->create_uvs_handle()),
		m_matIndexHdl(this->create_mat_index_handle()),
		m_pointsAttr(this->aquire(m_pointsHdl)),
		m_normalsAttr(this->aquire(m_normalsHdl)),
		m_uvsAttr(this->aquire(m_uvsHdl)),
		m_matIndexAttr(this->aquire(m_matIndexHdl))
	{}

	Polygons(const Polygons&) = delete;
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
			return { attrHandle, AttributeHelper<AttributeHandle>::add(m_vertexAttributes,
																	   m_faceAttributes, name, omAttr) };
		} else {
			// Found in OpenMesh already, now find it in our list
			auto opt = AttributeHelper<AttributeHandle>::find(m_vertexAttributes, m_faceAttributes, name);
			mAssertMsg(opt.has_value(), "This should never ever happen that we have a "
					   "property in OpenMesh but not in our custom list");
			return { attrHandle, opt.value() };
		}
	}

	template < class AttributeHandle >
	void remove(const AttributeHandle &attr) {
		// Remove from both lists
		m_meshData.remove_property(attr.omHandle);
		AttributeHelper<AttributeHandle>::remove(m_vertexAttributes, m_faceAttributes, attr.customHandle);
	}

	template < class AttributeHandle >
	std::optional<AttributeHandle> find(const std::string& name) {
		using Type = typename AttributeHandle::Type;
		using Attribute = typename AttributeHandle::Attribute;

		typename AttributeHandle::OmAttrHandle attrHandle;
		if(!m_meshData.get_property_handle(attrHandle, name))
			return std::nullopt;
		// Find attribute in custom list as 
		auto opt = AttributeHelper<AttributeHandle>::find(m_vertexAttributes, m_faceAttributes, name);
		mAssertMsg(opt.has_value(), "This should never ever happen that we have a "
				   "property in OpenMesh but not in our custom list");
		return { attrHandle, opt.value() };
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
	TriangleHandle add(const Triangle& tri, MaterialIndex idx);
	TriangleHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
					   MaterialIndex idx);
	TriangleHandle add(const std::array<VertexHandle, 3u>& vertices, MaterialIndex idx);
	// Adds a new quad.
	QuadHandle add(const Quad& quad, MaterialIndex idx);
	QuadHandle add(const VertexHandle& v0, const VertexHandle& v1, const VertexHandle& v2,
					   const VertexHandle& v3, MaterialIndex idx);
	QuadHandle add(const std::array<VertexHandle, 4u>& vertices, MaterialIndex idx);

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
	std::size_t add_bulk(Attribute<Type>& attribute, const VertexHandle& startVertex,
						 std::size_t count, std::istream& attrStream) {
		mAssert(startVertex.is_valid() && static_cast<std::size_t>(startVertex.idx()) < m_meshData.n_vertices());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData.n_vertices() - static_cast<std::size_t>(startVertex.idx()),
										   count);
		// Read the attribute from the stream
		attribute.restore(attrStream, static_cast<std::size_t>(startVertex.idx()), actualCount);
		std::size_t actuallyRead = static_cast<std::size_t>(attrStream.gcount()) / sizeof(Type);
		return actuallyRead;
	}
	/**
	 * Bulk-loads the given attribute starting at the given face.
	 * The number of read values will be capped by the number of faces present
	 * after the starting position.
	 */
	template < class Type >
	std::size_t add_bulk(Attribute<Type>& attribute, const FaceHandle& startFace,
						 std::size_t count, std::istream& attrStream) {
		mAssert(startFace.is_valid() && static_cast<std::size_t>(startFace.idx()) < m_meshData.n_faces());
		// Cap the number of attributes
		std::size_t actualCount = std::min(m_meshData.n_faces() - static_cast<std::size_t>(startFace.idx()),
										   count);
		// Read the attribute from the stream
		attribute.restore(attrStream, static_cast<std::size_t>(startFace.idx()), actualCount);
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

	const VertexAttributeHandle<OpenMesh::Vec3f>& get_points_handle() const {
		return m_pointsHdl;
	}

	const VertexAttributeHandle<OpenMesh::Vec3f>& get_normals_handle() const {
		return m_normalsHdl;
	}

	const VertexAttributeHandle<OpenMesh::Vec2f>& get_uvs_handle() const {
		return m_uvsHdl;
	}

	const FaceAttributeHandle<MaterialIndex>& get_mat_indices_handle() const {
		return m_matIndexHdl;
	}

	Attribute<MaterialIndex>& get_mat_indices() {
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
	// Helper struct for adding attributes since functions cannot be partially specialized
	template < class AttributeHandle >
	struct AttributeHelper;

	template < class T >
	struct AttributeHelper<VertexAttributeHandle<T>> {
		static AttributeHandle<T> add(AttributeListType& vertexList, AttributeListType& faceList, const std::string& name, OpenMesh::PropertyT<T>& prop) {
			return vertexList.add<T>(name, prop);
		}

		static std::optional<AttributeHandle<T>> find(AttributeListType& vertexList, AttributeListType& faceList, const std::string& name) {
			return vertexList.find<T>(name);
		}

		static void remove(AttributeListType& vertexList, AttributeListType& faceList, AttributeHandle<T>& hdl) {
			vertexList.remove(hdl);
		}
	};

	template < class T >
	struct AttributeHelper<FaceAttributeHandle<T>> {
		static AttributeHandle<T> add(AttributeListType& vertexList, AttributeListType& faceList, const std::string& name, OpenMesh::PropertyT<T>& prop) {
			return faceList.add<T>(name, prop);
		}

		static std::optional<AttributeHandle<T>> find(AttributeListType& vertexList, AttributeListType& faceList, const std::string& name) {
			return faceList.find<T>(name);
		}

		static void remove(AttributeListType& vertexList, AttributeListType& faceList, AttributeHandle<T>& hdl) {
			faceList.remove(hdl);
		}
	};

	// These methods simply create references to the attributes
	// TODO: by holding references to them, if they ever get removed, we're in a bad spot
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
	VertexAttributeHandle<OpenMesh::Vec3f> m_pointsHdl;
	VertexAttributeHandle<OpenMesh::Vec3f> m_normalsHdl;
	VertexAttributeHandle<OpenMesh::Vec2f> m_uvsHdl;
	FaceAttributeHandle<MaterialIndex> m_matIndexHdl;
	Attribute<OpenMesh::Vec3f>& m_pointsAttr;
	Attribute<OpenMesh::Vec3f>& m_normalsAttr;
	Attribute<OpenMesh::Vec2f>& m_uvsAttr;
	Attribute<MaterialIndex>& m_matIndexAttr;
};

} // namespace mufflon::scene::geometry