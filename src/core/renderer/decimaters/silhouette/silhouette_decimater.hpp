#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/scene/lod.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <cstddef>
#include <memory>

namespace mufflon::renderer::decimaters::silhouette {

class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;

	ImportanceDecimater(scene::Lod& original, scene::Lod& decimated,
						const Degrees maxNormalDeviation,
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
	void iterate(const std::size_t minVertexCount, const float reduction);

	// Methods for updating importance from trace events (only touch absolute importance)
	void record_silhouette_vertex_contribution(const u32 localIndex, const float importance);
	void record_shadow(const float irradiance);
	void record_direct_hit(const u32* vertexIndices, const u32 vertexCount,
						   const ei::Vec3& hitpoint, const float cosAngle);
	void record_direct_irradiance(const u32* vertexIndices, const u32 vertexCount,
								  const ei::Vec3& hitpoint, const float irradiance);
	void record_indirect_irradiance(const u32* vertexIndices, const u32 vertexCount,
									const ei::Vec3& hitpoint, const float irradiance);

	// Functions for querying internal state
	float get_current_max_importance() const;
	float get_current_importance(const u32 localFaceIndex, const ei::Vec3& hitpoint) const;
	float get_mapped_max_importance() const;
	float get_mapped_importance(const u32 originalFaceIndex, const ei::Vec3& hitpoint) const;
	double get_importance_sum() const noexcept { return m_importanceSum; }

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

	double m_importanceSum = 0.0;									// Stores the current importance sum (updates in update_importance_density)
	
	// General stuff
	std::unique_ptr<std::atomic<float>[]> m_importance;				// Importance hits (not light!)
	std::unique_ptr<std::atomic<float>[]> m_irradiance;				// Accumulated irradiance
	std::unique_ptr<std::atomic<u32>[]> m_hitCounter;				// Number of hits
	// Decimated mesh properties
	OpenMesh::VPropHandleT<VertexHandle> m_originalVertex;			// Vertex handle in the original mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_accumulatedImportanceDensity;	// Tracks the remapped importance (accumulated, if dampened)
	OpenMesh::VPropHandleT<VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh

	const Degrees m_maxNormalDeviation;

	std::atomic<float> m_shadowImportance;
	std::atomic<float> m_shadowSilhouetteImportance;
};

} // namespace mufflon::renderer::decimaters::silhouette