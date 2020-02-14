#pragma once

#include "animation_pt_params.hpp"
#include "util/string_view.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/lod.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/renderer/decimaters/silhouette/pt/silhouette_pt_common.hpp"
#include <deque>

namespace mufflon::renderer::decimaters::animation::pt {

class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;
	static constexpr Device DEVICE = Device::CPU;

	ImportanceDecimater(StringView objectName, ArrayDevHandle_t<DEVICE, silhouette::pt::Importances<DEVICE>> impBuffer,
						scene::Lod& original, scene::Lod& decimated,
						const std::size_t initialCollapses, const u32 frameCount,
						const float viewWeight, const float lightWeight,
						const float shadowWeight, const float shadowSilhouetteWeight);
	ImportanceDecimater(const ImportanceDecimater&) = delete;
	ImportanceDecimater(ImportanceDecimater&&);
	ImportanceDecimater& operator=(const ImportanceDecimater&) = delete;
	ImportanceDecimater& operator=(ImportanceDecimater&&) = delete;
	~ImportanceDecimater();

	// Resizes the buffers properly
	ArrayDevHandle_t<DEVICE, silhouette::pt::Importances<DEVICE>> start_iteration();
	// Updates the importance densities of the decimated mesh
	void update_importance_density(const silhouette::pt::ImportanceSums& impSums);
	// Uploads the importance with a given function for inter-frame coherency
	void upload_importance(const PImpWeightMethod::Values weighting, u32 frameStart, u32 frameEnd);
	/* Updates the decimated mesh by collapsing and uncollapsing vertices.
	 * The specified threshold determines when a vertex collapses or gets restored
	 */
	void iterate(const std::size_t targetCount);

	// Functions for querying internal state
	float get_current_max_importance() const;
	double get_importance_sum(const std::size_t index) const noexcept { return m_importanceSum[index]; }
	float get_accumulated_importance(const scene::geometry::PolygonMeshType::VertexHandle vertex) {
		return m_originalMesh.property(m_accumulatedImportanceDensity, vertex);
	}

	std::size_t get_original_vertex_count() const noexcept;
	std::size_t get_decimated_vertex_count() const noexcept;

private:
	// Returns the vertex handle in the original mesh
	VertexHandle get_original_vertex_handle(const VertexHandle decimatedHandle) const;
	void decimate_with_error_quadrics(const std::size_t collapses);

	// Recomputes normals for decimated mesh
	void recompute_geometric_vertex_normals();

	StringView m_objectName;
	scene::Lod& m_original;
	scene::Lod& m_decimated;
	scene::geometry::Polygons& m_originalPoly;
	scene::geometry::Polygons* m_decimatedPoly;
	Mesh& m_originalMesh;
	Mesh* m_decimatedMesh;
	u32 m_frameCount;

	std::vector<double> m_importanceSum{};									// Stores the current importance sum (updates in update_importance_density)

	// General stuff
	// Each mesh requires an importance buffer per animation frame in the sliding window
	ArrayDevHandle_t<DEVICE, silhouette::pt::Importances<DEVICE>> m_importanceBuffer;
	ArrayDevHandle_t<DEVICE, silhouette::pt::Importances<DEVICE>> m_currImpBuffer;		// Slice within importanceBuffer for current iteration
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

} // namespace mufflon::renderer::decimaters::animation::pt