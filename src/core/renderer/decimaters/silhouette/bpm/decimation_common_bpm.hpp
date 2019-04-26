#pragma once

#include "silhouette_bpm_common.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"

namespace mufflon::renderer::decimaters::silhouette::bpm {

template < Device dev >
class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;
	static constexpr Device DEVICE = dev;

	ImportanceDecimater(scene::Lod& original, scene::Lod& decimated,
						   const std::size_t initialCollapses,
						   const float viewWeight, const float lightWeight,
						   const float shadowWeight, const float shadowSilhouetteWeight);
	ImportanceDecimater(const ImportanceDecimater&) = delete;
	ImportanceDecimater(ImportanceDecimater&&);
	ImportanceDecimater& operator=(const ImportanceDecimater&) = delete;
	ImportanceDecimater& operator=(ImportanceDecimater&&) = delete;
	~ImportanceDecimater();

	void copy_back_normalized_importance();

	// Resizes the buffers properly
	ArrayDevHandle_t<dev, Importances<dev>> start_iteration();
	// Updates the importance densities of the decimated mesh
	void update_importance_density(const ImportanceSums& impSums);
	/* Updates the decimated mesh by collapsing and uncollapsing vertices.
	 * The specified threshold determines when a vertex collapses or gets restored
	 */
	void iterate(const std::size_t minVertexCount, const float reduction);

	// Functions for querying internal state
	float get_current_max_importance() const;
	double get_importance_sum() const noexcept { return m_importanceSum; }

	std::size_t get_original_vertex_count() const noexcept;
	std::size_t get_decimated_vertex_count() const noexcept;

private:
	// Returns the vertex handle in the original mesh
	VertexHandle get_original_vertex_handle(const VertexHandle decimatedHandle) const;
	void decimate_with_error_quadrics(const std::size_t collapses);
	void pull_importance_from_device();

	// Recomputes normals for decimated mesh
	void recompute_geometric_vertex_normals();

	scene::Lod& m_original;
	scene::Lod& m_decimated;
	scene::geometry::Polygons& m_originalPoly;
	scene::geometry::Polygons* m_decimatedPoly;
	Mesh& m_originalMesh;
	Mesh* m_decimatedMesh;

	double m_importanceSum = 0.0;									// Stores the current importance sum (updates in update_importance_density)

	// General stuff
	unique_device_ptr<dev, Importances<dev>[]> m_devImportances;	// Buffer on the device
	std::unique_ptr<Importances<dev>[]> m_importances;				// Buffer for copying from device
	// Decimated mesh properties
	OpenMesh::VPropHandleT<VertexHandle> m_originalVertex;			// Vertex handle in the original mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_accumulatedImportanceDensity;	// Tracks the remapped importance (accumulated, if dampened)
	OpenMesh::VPropHandleT<VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh

	const float m_viewWeight;										// Weight assigned to the viewpath importance
	const float m_lightWeight;										// Weight assigned to the irradiance-based importance
	const float m_shadowWeight;										// Weight assigned to the shadow importance (sum only)
	const float m_shadowSilhouetteWeight;							// Weight assigned to the shadow silhouette importance
};

} // namespace mufflon::renderer::decimaters::silhouette::bpm