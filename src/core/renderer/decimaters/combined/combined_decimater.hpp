#pragma once

#include "combined_params.hpp"
#include "util/string_view.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/renderer/decimaters/util/octree.hpp"

// Forward declarations
namespace mufflon::scene {
class Lod;
namespace geometry {
class Polygons;
} // namespace geometry
} // namespace mufflon::scene

namespace mufflon::renderer::decimaters::combined {

class CombinedDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;
	using VertexHandle = typename Mesh::VertexHandle;

	CombinedDecimater(StringView objectName, scene::Lod& original,
					  scene::Lod& decimated, const u32 frameCount,
					  ArrayDevHandle_t<Device::CPU, FloatOctree*> view,
					  ArrayDevHandle_t<Device::CPU, SampleOctree*> irradiance,
					  ArrayDevHandle_t<Device::CPU, double> importanceSums,
					  const float lightWeight);
	CombinedDecimater(const CombinedDecimater&) = delete;
	CombinedDecimater(CombinedDecimater&&);
	CombinedDecimater& operator=(const CombinedDecimater&) = delete;
	CombinedDecimater& operator=(CombinedDecimater&&) = delete;
	~CombinedDecimater();
	
	void finish_gather(const u32 frame);
	void update(const PImpWeightMethod::Values weighting,
				u32 startFrame, u32 endFrame);
	void reduce(const std::size_t targetVertexCount);

	StringView get_mesh_name() const noexcept { return m_objectName; }
	FloatOctree& get_view_octree(const u32 frame) noexcept { return *m_viewImportance[frame]; }
	SampleOctree& get_irradiance_octree(const u32 frame) noexcept { return *m_irradianceImportance[frame]; }
	double get_importance_sum(const u32 frame) const noexcept { return m_importanceSums[frame]; }
	std::size_t get_original_vertex_count() const noexcept;
	std::size_t get_decimated_vertex_count() const noexcept;

private:
	VertexHandle get_original_vertex_handle(const VertexHandle decimatedHandle) const noexcept;

	StringView m_objectName;
	scene::Lod& m_original;
	scene::Lod& m_decimated;
	scene::geometry::Polygons& m_originalPoly;
	scene::geometry::Polygons* m_decimatedPoly;
	Mesh& m_originalMesh;
	Mesh* m_decimatedMesh;

	// Importance octrees
	ArrayDevHandle_t<Device::CPU, FloatOctree*> m_viewImportance;
	ArrayDevHandle_t<Device::CPU, SampleOctree*> m_irradianceImportance;
	ArrayDevHandle_t<Device::CPU, double> m_importanceSums;			// Stores the importance sum per frame

	const u32 m_frameCount;											// Total frame count in the animation sequence

	// Decimated mesh properties (TODO: necessary?)
	OpenMesh::VPropHandleT<VertexHandle> m_originalVertex;			// Vertex handle in the original mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_accumulatedImportanceDensity;	// Tracks the remapped importance (accumulated, if dampened)
	OpenMesh::VPropHandleT<VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh

	const float m_lightWeight;										// Weight assigned to the irradiance-based importance
};

} // namespace mufflon::renderer::decimaters::combined