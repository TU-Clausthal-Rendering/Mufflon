#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/vector.hpp>

namespace mufflon::scene::tessellation {

// Serves as an oracle for the tessellation level
class TessLevelOracle {
public:
	TessLevelOracle() = default;

	// Get the outer tessellation level (ie. the number of new vertices) for the given edge of the given face
	virtual u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const = 0;

	// Get the inner tessellation level (ie. the number of new vertices) for the given face
	virtual u32 get_inner_tessellation_level(const OpenMesh::FaceHandle face) const = 0;

	void set_mesh(geometry::PolygonMeshType* mesh) {
		m_mesh = mesh;
	}

protected:
	geometry::PolygonMeshType* m_mesh = nullptr;
};

class Tessellater {
public:
	Tessellater(TessLevelOracle& oracle) :
		m_tessLevelOracle(oracle) {}
	Tessellater(const Tessellater&) = delete;
	Tessellater(Tessellater&&) = delete;
	Tessellater& operator=(const Tessellater&) = delete;
	Tessellater& operator=(Tessellater&&) = delete;
	~Tessellater() = default;

	void set_phong_tessellation(bool enabled) {
		m_usePhongTessellation = enabled;
	}
	void tessellate(geometry::PolygonMeshType& mesh);

protected:
	// Stores the number of added vertices per edge. Also stores the offset into
	// m_edgeVertexHandles. Also stores the orientation of the edge for which
	// the vertices were added
	struct AddedVertices {
		OpenMesh::VertexHandle from;
		OpenMesh::VertexHandle to;
		u32 offset;
		u32 count;
	};

	// Set the vertex properties (position, normals etc.) for the newly created outer vertex.
	// The coordinate x is the linear coordinate between the two edge vertices
	virtual void set_edge_vertex(const float x,
								 const OpenMesh::EdgeHandle edge,
								 const OpenMesh::VertexHandle vertex);

	// Set the vertex properties (position, normals etc.) for the newly created inner vertex.
	// The coordinates x and y are the bilinear coordinates of the vertex with respect to the
	// first edge chosen for tessellation
	virtual void set_quad_inner_vertex(const float x, const float y,
									   const OpenMesh::VertexHandle vertex,
									   const OpenMesh::FaceHandle face,
									   const OpenMesh::VertexHandle(&vertices)[4u]);

	// Set the vertex properties (position, normals etc.) for the newly created inner vertex.
	// The coordinates x and y are the barycentric coordinates of the vertex with respect to the
	// first edge chosen for tessellation
	virtual void set_triangle_inner_vertex(const float x, const float y,
										   const OpenMesh::VertexHandle vertex,
										   const OpenMesh::FaceHandle face,
										   const OpenMesh::VertexHandle(&vertices)[4u]);

	// Set the face properties (material index etc.) for the newly created inner face
	virtual void set_quad_face_inner(const OpenMesh::FaceHandle original,
									 const OpenMesh::FaceHandle newInner);

	// Set the face properties (material index etc.) for the newly created outer face
	virtual void set_quad_face_outer(const OpenMesh::FaceHandle original,
									 const OpenMesh::FaceHandle newOuter);

	// Set the face properties (material index etc.) for the newly created inner face
	virtual void set_triangle_face_inner(const OpenMesh::FaceHandle original,
										 const OpenMesh::FaceHandle newOuter);

	// Set the face properties (material index etc.) for the newly created outer face
	virtual void set_triangle_face_outer(const OpenMesh::FaceHandle original,
										 const OpenMesh::FaceHandle newOuter);

	// Perfoms tessellation for the inner face (quad)
	virtual void tessellate_inner_quads(const u32 innerLevel, const OpenMesh::FaceHandle original);
	
	// Performs tessellation for the inner face (triangle)
	virtual void tessellate_inner_triangles(const u32 innerLevel, const OpenMesh::FaceHandle original);

	// Triangulate the given strip between inner and outer vertices
	virtual void triangulate_strip(const u32 lengthOuter, const u32 lengthInner,
								   const OpenMesh::FaceHandle original);

	virtual void pre_tessellate() { (void)0; }
	virtual void post_tessellate() { (void)0; }

	// Mesh to be tessellated
	geometry::PolygonMeshType* m_mesh = nullptr;
	// Handle or mesh property storing the offset and count of edge vertices
	// Offset indexes into m_edgeVertexHandles
	OpenMesh::EPropHandleT<AddedVertices> m_addedVertexProp;

	bool m_usePhongTessellation = false;

private:
	void spawn_inner_quad_vertices(const u32 innerLevel,
								   const OpenMesh::FaceHandle face,
								   const OpenMesh::VertexHandle(&vertices)[4u]);
	void spawn_inner_triangle_vertices(const u32 innerLevel,
									   const OpenMesh::FaceHandle face,
									   const OpenMesh::VertexHandle(&vertices)[4u]);
	// Spawns the quads between inner and outer vertices
	u32 spawn_outer_quads(const u32 innerLevel, const u32 outerLevel,
						  const u32 startInner, const u32 startOuter,
						  const u32 edgeVertexOffset, const u32 edgeIndex,
						  const bool swapEdgeVertices, const OpenMesh::FaceHandle face);
	// Spawns the triangles needed in the corners after quads have been added
	void spawn_outer_corner_triangles(const u32 innerLevel, const u32 startInner,
									  const u32 startOuter, const u32 outerQuadCount,
									  const u32 edgeIndex, const AddedVertices& outerVertices,
									  const OpenMesh::VertexHandle from,
									  const OpenMesh::VertexHandle to,
									  const OpenMesh::FaceHandle face);

	// Holds all vertices spawned for edges of the mesh
	std::vector<OpenMesh::VertexHandle> m_edgeVertexHandles;
	// Holds the inner tessellation vertices of the face currently being tessellated
	std::vector<OpenMesh::VertexHandle> m_innerVertices;
	// Holds the vertices of a face between outer and inner tessellation vertices
	std::vector<OpenMesh::VertexHandle> m_stripVertices;

	TessLevelOracle& m_tessLevelOracle;
};

} // mufflon::scene::tessellation