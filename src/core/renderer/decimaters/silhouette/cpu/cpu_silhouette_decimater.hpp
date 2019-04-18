#pragma once

#include "core/renderer/decimaters/silhouette/sil_common.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/scene/lod.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <cstddef>
#include <memory>

namespace mufflon::renderer::decimaters::silhouette {

class CpuImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;

	CpuImportanceDecimater(scene::Lod& original, scene::Lod& decimated,
						const std::size_t initialCollapses,
						const float viewWeight, const float lightWeight,
						const float shadowWeight, const float shadowSilhouetteWeight);
	CpuImportanceDecimater(const CpuImportanceDecimater&) = delete;
	CpuImportanceDecimater(CpuImportanceDecimater&&);
	CpuImportanceDecimater& operator=(const CpuImportanceDecimater&) = delete;
	CpuImportanceDecimater& operator=(CpuImportanceDecimater&&) = delete;
	~CpuImportanceDecimater();

	void upload_normalized_importance();

	// Resizes the buffers properly
	Importances<Device::CPU>* start_iteration();
	// Updates the importance densities of the decimated mesh
	void udpate_importance_density(const DeviceImportanceSums<Device::CPU>& impSums);
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
	unique_device_ptr<Device::CPU, Importances<Device::CPU>[]> m_devImportances;	// Importance values per vertex
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

} // namespace mufflon::renderer::decimaters::silhouette