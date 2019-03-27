#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/scene/lod.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <cstddef>
#include <memory>

namespace mufflon::renderer::silhouette::decimation {

// Specify what the collapse mode should be
enum class CollapseMode {
	DEFAULT,						// Default: no restriction
	NO_CONCAVE,						// Disallow concave collapses altogether
	NO_SILHOUETTE,					// Disallow silhouette vertices from collapsing away or onto
	DAMPENED_IMPORTANCE				// Importance from later passes gets weighted down
};

class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;

	ImportanceDecimater(scene::Lod& original, scene::Lod& decimated,
						const Degrees maxNormalDeviation,
						const CollapseMode mode,
						const std::size_t initialCollapses);
	ImportanceDecimater(const ImportanceDecimater&) = delete;
	ImportanceDecimater(ImportanceDecimater&&);
	ImportanceDecimater& operator=(const ImportanceDecimater&) = delete;
	ImportanceDecimater& operator=(ImportanceDecimater&&) = delete;
	~ImportanceDecimater();

	// Updates the importance densities of the decimated mesh
	void udpate_importance_density();
	/* Updates the decimated mesh by collapsing and uncollapsing vertices.
	 * The specified threshold determines when a vertex collapses or gets restored
	 */
	void iterate(const std::size_t minVertexCount, const float threshold, const float reduction);

	// Methods for updating importance from trace events (only touch absolute importance)
	// Increases the importance of a vertex
	void record_silhouette_vertex_contribution(const u32 localIndex, const float importance);
	// Increases the importance for all bordering vertices of a face, weighted by distance² to the hit point
	void record_face_contribution(const u32* vertexIndices, const u32 vertexCount,
								  const ei::Vec3& hitpoint, const float importance);

	float get_max_importance() const;
	float get_max_importance_density() const;
	float get_importance(const u32 localFaceIndex, const ei::Vec3& hitpoint) const;
	float get_importance_density(const u32 localFaceIndex, const ei::Vec3& hitpoint) const;

	std::size_t get_original_vertex_count() const noexcept;
	std::size_t get_decimated_vertex_count() const noexcept;

private:
	// Returns the vertex handle in the original mesh
	VertexHandle get_original_vertex_handle(const VertexHandle decimatedHandle) const;

	scene::Lod& m_original;
	scene::Lod& m_decimated;
	scene::geometry::Polygons& m_originalPoly;
	scene::geometry::Polygons* m_decimatedPoly;
	Mesh& m_originalMesh;
	Mesh* m_decimatedMesh;
	
	// General stuff
	std::unique_ptr<std::atomic<float>[]> m_importance;				// Absolute importance per vertex (accumulative, indexed in decimated vertex handles!)
	// Decimated mesh properties
	OpenMesh::VPropHandleT<VertexHandle> m_originalVertex;			// Vertex handle in the original mesh
	OpenMesh::VPropHandleT<float> m_importanceDensity;				// Importance per m² for the decimated mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh
	OpenMesh::VPropHandleT<bool> m_silhouette;						// Was this vertex ever detected as a shadow silhouette

	const Degrees m_maxNormalDeviation;
	const CollapseMode m_collapseMode;								// Specify how to deal with concave collapses

	float m_damping = 1.f;											// Current factor for importance
};

} // namespace mufflon::renderer::silhouette::decimation