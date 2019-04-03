#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/scene/lod.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <cstddef>
#include <memory>

namespace mufflon::renderer::decimaters::importance {

class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;

	ImportanceDecimater(scene::Lod& original, scene::Lod& decimated,
						const std::size_t initialCollapses,
						const Degrees maxNormalDeviation,
						const float viewWeight, const float lightWeight);
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

	void record_direct_hit(const u32* vertexIndices, const u32 vertexCount,
						   const ei::Vec3& hitpoint, const float cosAngle,
						   const float sharpness);
	void record_direct_irradiance(const u32* vertexIndices, const u32 vertexCount,
								  const ei::Vec3& hitpoint, const float irradiance);
	void record_indirect_irradiance(const u32* vertexIndices, const u32 vertexCount,
									const ei::Vec3& hitpoint, const float irradiance);

	float get_current_max_importance() const;
	float get_current_importance(const u32 localFaceIndex, const ei::Vec3& hitpoint) const;
	float get_mapped_max_importance() const;
	float get_mapped_importance(const u32 originalFaceIndex, const ei::Vec3& hitpoint) const;
	double get_importance_sum() const noexcept { return m_importanceSum; }

	std::size_t get_original_vertex_count() const noexcept;
	std::size_t get_decimated_vertex_count() const noexcept;

private:
	struct Importances {
		std::atomic<float> viewImportance;	// Importance hits (not light!); also holds final normalized importance value after update
		std::atomic<float> irradiance;		// Accumulated irradiance
		std::atomic<u32> hitCounter;		// Number of hits
	};

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
	std::unique_ptr<Importances[]> m_importances;					// Importance values per vertex
	// Decimated mesh properties
	OpenMesh::VPropHandleT<VertexHandle> m_originalVertex;			// Vertex handle in the original mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_importanceDensity;				// Mapped importance in the original mesh
	OpenMesh::VPropHandleT<VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh

	const Degrees m_maxNormalDeviation;								// Maximum allowed normal deviation after collapse

	const float m_viewWeight;										// Weight assigned to the viewpath importance
	const float m_lightWeight;										// Weight assigned to the irradiance-based importance
};

} // namespace mufflon::renderer::decimaters::importance