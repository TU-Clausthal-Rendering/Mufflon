#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/vector.hpp>

namespace mufflon::scene::tessellation {

class AdaptiveTessellater {
public:
	struct AddedVertices {
		u32 offset;
		u32 count;
	};

	AdaptiveTessellater(geometry::PolygonMeshType& mesh);
	AdaptiveTessellater(const AdaptiveTessellater&) = delete;
	AdaptiveTessellater(AdaptiveTessellater&&) = delete;
	AdaptiveTessellater& operator=(const AdaptiveTessellater&) = delete;
	AdaptiveTessellater& operator=(AdaptiveTessellater&&) = delete;
	~AdaptiveTessellater() = default;

	bool tessellate();

protected:
	// Get the outer tessellation level (ie. the number of new vertices) for the given edge of the given face
	virtual u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const;

	// Get the inner tessellation level (ie. the number of new vertices) for the given face
	virtual u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const;

	// Set the vertex properties (position, normals etc.) for the newly created outer vertex
	virtual void set_edge_vertex(const u32 i, const u32 count,
								 const OpenMesh::EdgeHandle edge,
								 const OpenMesh::VertexHandle vertex);

	// Set the vertex properties (position, normals etc.) for the newly created inner vertex
	virtual void set_quad_inner_vertex(const u32 x, const u32 y, const u32 level,
									   const OpenMesh::VertexHandle vertex,
									   const OpenMesh::FaceHandle face,
									   const OpenMesh::VertexHandle(&vertices)[4u]);

	// Set the face properties (material index etc.) for the newly created inner face
	virtual void set_quad_face_inner(const u32 x, const u32 y, const u32 innerLevel,
									 const OpenMesh::FaceHandle original,
									 const OpenMesh::FaceHandle newInner);

	// Set the face properties (material index etc.) for the newly created outer face
	virtual void set_quad_face_outer(const OpenMesh::FaceHandle original,
									 const OpenMesh::FaceHandle newOuter);

	// Set the face properties (material index etc.) for the newly created outer face
	virtual void set_triangle_face_outer(const OpenMesh::FaceHandle original,
										 const OpenMesh::FaceHandle newOuter);

	// Perfoms tessellation for the inner face (quad)
	virtual void tessellate_inner_quads(const u32 innerLevel, const OpenMesh::FaceHandle original);

	geometry::PolygonMeshType& m_mesh;
	OpenMesh::EPropHandleT<AddedVertices> m_addedVertexProp;

private:
	std::vector<OpenMesh::VertexHandle> m_edgeVertexHandles;
	std::vector<OpenMesh::VertexHandle> m_innerVertices;
	std::vector<OpenMesh::VertexHandle> m_stripVertices;
};

} // mufflon::scene::tessellation