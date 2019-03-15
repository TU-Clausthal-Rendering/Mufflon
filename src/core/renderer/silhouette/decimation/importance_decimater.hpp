#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/renderer/silhouette/sil_imp_map.hpp"
#include "core/scene/lod.hpp"
#include <OpenMesh/Tools/Utils/HeapT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <cstddef>
#include <memory>

namespace mufflon::renderer::silhouette::decimation {

class ImportanceDecimater {
public:
	using Mesh = scene::geometry::PolygonMeshType;

	ImportanceDecimater(scene::Lod& original, scene::Lod& decimated, const std::size_t initialCollapses);
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
	void iterate(const std::size_t minVertexCount, const float threshold);

	// Methods for updating importance from trace events (only touch absolute importance)
	// Increases the importance of a vertex
	void record_vertex_contribution(const u32 localIndex, const float importance);
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
	using CollapseInfo = OpenMesh::Decimater::CollapseInfoT<Mesh>;

	// Interface for the vertex heap
	class HeapInterface {
	public:
		HeapInterface(Mesh& mesh, OpenMesh::VPropHandleT<float> priority,
					  OpenMesh::VPropHandleT<int> heapPosition) :
			m_mesh(mesh),
			m_priority(priority),
			m_heapPosition(heapPosition) {}

		bool less(Mesh::VertexHandle vh0, Mesh::VertexHandle vh1) {
			return m_mesh.property(m_priority, vh0) < m_mesh.property(m_priority, vh1);
		}

		bool greater(Mesh::VertexHandle vh0, Mesh::VertexHandle vh1) {
			return m_mesh.property(m_priority, vh0) > m_mesh.property(m_priority, vh1);
		}

		int get_heap_position(Mesh::VertexHandle vh) {
			return m_mesh.property(m_heapPosition, vh);
		}

		void set_heap_position(Mesh::VertexHandle vh, int pos) {
			m_mesh.property(m_heapPosition, vh) = pos;
		}


	private:
		Mesh& m_mesh;
		OpenMesh::VPropHandleT<float> m_priority;
		OpenMesh::VPropHandleT<int> m_heapPosition;
	};

	// Necessary history to store per-vertex to reinsert
	struct CollapseHistory {
		Mesh::VertexHandle v1;
		Mesh::VertexHandle vl;
		Mesh::VertexHandle vr;
	};

	// Returns the vertex handle in the original mesh
	Mesh::VertexHandle get_original_vertex_handle(const Mesh::VertexHandle decimatedHandle) const;

	std::size_t collapse(const float threshold);
	std::size_t uncollapse(const float threshold);

	// Computes the new importance densities
	float compute_new_importance_densities(std::vector<std::pair<float, float>>& newDensities,
										   const Mesh::VertexHandle v0, const Mesh::VertexHandle v1,
										   const Mesh::VertexHandle vl, const Mesh::VertexHandle vr,
										   const float threshold) const;
	void add_vertex_collapse(const Mesh::VertexHandle vertex, const float threshold);
	bool is_collapse_legal(const OpenMesh::Decimater::CollapseInfoT<Mesh>& ci) const;
	float collapse_priority(const OpenMesh::Decimater::CollapseInfoT<Mesh>& ci) const;

	scene::Lod& m_original;
	scene::Lod& m_decimated;
	scene::geometry::Polygons& m_originalPoly;
	scene::geometry::Polygons& m_decimatedPoly;
	Mesh& m_originalMesh;
	Mesh& m_decimatedMesh;
	
	// General stuff
	std::unique_ptr<std::atomic<float>[]> m_importance;				// Absolute importance per vertex (accumulative, indexed in original vertex handles!)
	// Decimated mesh properties
	OpenMesh::VPropHandleT<Mesh::VertexHandle> m_originalVertex;	// Vertex handle in the original mesh
	OpenMesh::VPropHandleT<float> m_importanceDensity;				// Importance per m² for the decimated mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<Mesh::VertexHandle> m_collapsedTo;		// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;						// Whether collapsedTo refers to original or decimated mesh

	// Stuff for decimation
	std::unique_ptr<OpenMesh::Utils::HeapT<Mesh::VertexHandle, HeapInterface>> m_heap;
	OpenMesh::VPropHandleT<Mesh::HalfedgeHandle> m_collapseTarget;	// Stores the collapse target halfedge for a vertex
	OpenMesh::VPropHandleT<float> m_priority;						// Stores the collapse priority for a vertex and its target
	OpenMesh::VPropHandleT<int> m_heapPosition;						// Position of vertex in the heap
};

// Used in initial decimation
template < class MeshT = scene::geometry::PolygonMeshType >
class DecimationTrackerModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(DecimationTrackerModule, MeshT, DecimationTrackerModule);

	DecimationTrackerModule(MeshT& mesh);
	virtual ~DecimationTrackerModule() = default;
	DecimationTrackerModule(const DecimationTrackerModule&) = delete;
	DecimationTrackerModule(DecimationTrackerModule&&) = delete;
	DecimationTrackerModule& operator=(const DecimationTrackerModule&) = delete;
	DecimationTrackerModule& operator=(DecimationTrackerModule&&) = delete;

	void set_properties(MeshT& originalMesh, OpenMesh::VPropHandleT<typename MeshT::VertexHandle> originalVertex,
						OpenMesh::VPropHandleT<typename MeshT::VertexHandle> collapsedTo,
						OpenMesh::VPropHandleT<bool> collapsed);
	float collapse_priority(const CollapseInfo& ci) final;
	void postprocess_collapse(const CollapseInfo& ci) final;

private:
	MeshT* m_originalMesh;

	OpenMesh::VPropHandleT<typename MeshT::VertexHandle> m_originalVertex;	// Vertex handle in the original mesh
	// Original mesh properties
	OpenMesh::VPropHandleT<typename MeshT::VertexHandle> m_collapsedTo;		// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
	OpenMesh::VPropHandleT<bool> m_collapsed;								// Whether collapsedTo refers to original or decimated mesh
};

} // namespace mufflon::renderer::silhouette::decimation