#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/scene/types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <ei/vector.hpp>
#include <tuple>
#include <vector>

namespace mufflon::scene {

class Scenario;

namespace tessellation {

constexpr float PHONGTESS_ALPHA = 0.5f;

// Serves as an oracle for the tessellation level
class TessLevelOracle {
public:
	TessLevelOracle() = default;

	// Get the outer tessellation level (ie. the number of new vertices) for the given edge of the given face
	virtual u32 get_edge_tessellation_level(const OpenMesh::EdgeHandle edge) const = 0;

	virtual u32 get_triangle_inner_tessellation_level(const OpenMesh::FaceHandle face) const = 0;
	virtual std::pair<u32, u32> get_quad_inner_tessellation_level(const OpenMesh::FaceHandle face) const = 0;

	void set_mesh(geometry::PolygonMeshType* mesh) {
		m_mesh = mesh;
	}

	void set_phong_tessellation(bool enabled) {
		m_usePhongTessellation = enabled;
	}

	void set_mat_properties(const Scenario& scenario, OpenMesh::FPropHandleT<MaterialIndex> matHdl) noexcept {
		m_scenario = &scenario;
		m_matHdl = matHdl;
	}

protected:
	geometry::PolygonMeshType* m_mesh = nullptr;
	bool m_usePhongTessellation = false;
	OpenMesh::FPropHandleT<MaterialIndex> m_matHdl;
	const Scenario* m_scenario = nullptr;
};

class Tessellater {
public:
	Tessellater(TessLevelOracle& oracle) noexcept :
		m_tessLevelOracle(oracle) {}
	Tessellater(const Tessellater&) = delete;
	Tessellater(Tessellater&&) = delete;
	Tessellater& operator=(const Tessellater&) = delete;
	Tessellater& operator=(Tessellater&&) = delete;
	~Tessellater() = default;

	void set_phong_tessellation(bool enabled) {
		m_usePhongTessellation = enabled;
	}

	// Returns the handle of the temporary face used for property copies.
	// Doesn't garbage collect. Also stores the old face handle for all new faces (get_old_face_property()).
	OpenMesh::FaceHandle tessellate(geometry::PolygonMeshType& mesh);

	OpenMesh::FPropHandleT<OpenMesh::FaceHandle> get_old_face_property() const noexcept {
		return m_oldFace;
	}

protected:
	// Stores the number of added vertices per edge. Also stores the offset into
	// m_edgeVertexHandles. Also stores the orientation of the edge for which
	// the vertices were added
	struct AddedVertices {
		OpenMesh::VertexHandle from;
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
									   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices);

	// Set the vertex properties (position, normals etc.) for the newly created inner vertex.
	// The coordinates x and y are the barycentric coordinates of the vertex with respect to the
	// first edge chosen for tessellation
	virtual void set_triangle_inner_vertex(const float x, const float y,
										   const OpenMesh::VertexHandle vertex,
										   const OpenMesh::FaceHandle face,
										   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices);

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
	virtual void tessellate_inner_quads(const u32 innerLevelX, const u32 innerLevelY,
										const OpenMesh::FaceHandle original);

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
	// Keeps track of inserted faces/vertices
	OpenMesh::FPropHandleT<OpenMesh::FaceHandle> m_oldFace;

	bool m_usePhongTessellation = false;

private:
	void tessellate_edges();
	void tessellate_triangle(const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices,
							 const OpenMesh::FaceHandle face, const u32 innerLevel);
	void tessellate_quad(const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices,
						 const OpenMesh::FaceHandle face, const u32 innerLevelX, const u32 innerLevelY);

	void spawn_inner_quad_vertices(const u32 innerLevelX, const u32 innerLevelY,
								   const OpenMesh::FaceHandle face,
								   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices);
	void spawn_inner_triangle_vertices(const u32 innerLevel,
									   const OpenMesh::FaceHandle face,
									   const std::vector<std::pair<OpenMesh::VertexHandle, AddedVertices>>& vertices);
	// Spawns the quads between inner and outer vertices
	u32 spawn_outer_quads(const u32 innerLevelX, const u32 innerLevelY,
						  const u32 currInnerLevel, const u32 outerLevel,
						  const u32 startInner, const u32 startOuter,
						  const u32 edgeVertexOffset, const u32 edgeIndex,
						  const bool swapEdgeVertices, const OpenMesh::FaceHandle face);
	// Spawns the triangles needed in the corners after quads have been added
	void spawn_outer_corner_triangles(const u32 innerLevelX, const u32 innerLevelY, const u32 currInnerLevel,
									  const u32 startInner, const u32 startOuter, const u32 outerQuadCount,
									  const u32 edgeIndex, const AddedVertices& outerVertices,
									  const OpenMesh::VertexHandle from,
									  const OpenMesh::VertexHandle to,
									  const OpenMesh::FaceHandle face,
									  bool doLeft, bool doRight);
	// Tesssellates the border region between inner quads and edge if good-looking quads might be possible
	void spawn_outer_quad_corner_region(const u32 innerLevelX, const u32 innerLevelY, const u32 currInnerLevel,
										const u32 otherInnerLevel, const u32 startInner, const u32 startOuter,
										const bool edgeNeedsFlip, const u32 edgeIndex, const u32 outerQuadCount,
										const OpenMesh::VertexHandle from, const OpenMesh::VertexHandle to,
										const OpenMesh::VertexHandle prevFrom,
										const OpenMesh::FaceHandle face, const AddedVertices& outerVertices,
										const AddedVertices& prevOuterVertices, const AddedVertices& nextOuterVertices);

	OpenMesh::VertexHandle get_inner_vertex_triangle(const u32 edgeIndex, const u32 index, const u32 innerLevelX) const;
	OpenMesh::VertexHandle get_inner_vertex_quad(const u32 edgeIndex, const u32 index,
												 const u32 innerLevelX, const u32 innerLevelY) const;

	// Holds all vertices spawned for edges of the mesh
	std::vector<OpenMesh::VertexHandle> m_edgeVertexHandles;
	// Holds the inner tessellation vertices of the face currently being tessellated
	std::vector<OpenMesh::VertexHandle> m_innerVertices;
	// Holds the vertices of a face between outer and inner tessellation vertices
	std::vector<OpenMesh::VertexHandle> m_stripVertices;

	TessLevelOracle& m_tessLevelOracle;
};

} // namespace tessellation
} // namespace mufflon::scene