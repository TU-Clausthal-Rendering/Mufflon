#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::decimaters::modules {

// Tracks the collapses of a mesh
template < class MeshT = scene::geometry::PolygonMeshType >
class CollapseTrackerModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(CollapseTrackerModule, MeshT, CollapseTrackerModule);

	CollapseTrackerModule(MeshT& mesh) :
		Base(mesh, true) {}
	virtual ~CollapseTrackerModule() = default;
	CollapseTrackerModule(const CollapseTrackerModule&) = delete;
	CollapseTrackerModule(CollapseTrackerModule&&) = delete;
	CollapseTrackerModule& operator=(const CollapseTrackerModule&) = delete;
	CollapseTrackerModule& operator=(CollapseTrackerModule&&) = delete;

	void set_properties(MeshT& originalMesh, OpenMesh::VPropHandleT<bool> collapsed,
						OpenMesh::VPropHandleT<typename MeshT::VertexHandle> collapsedTo) {
		m_originalMesh = &originalMesh;
		m_collapsed = collapsed;
		m_collapsedTo = collapsedTo;
	}

	void postprocess_collapse(const CollapseInfo& ci) final {
		// Assumes that the two meshes are initially equal
		m_originalMesh->property(m_collapsed, ci.v0) = true;
		m_originalMesh->property(m_collapsedTo, ci.v0) = ci.v1;
	}

private:
	MeshT* m_originalMesh;

	// Original mesh properties
	OpenMesh::VPropHandleT<bool> m_collapsed;										// Whether collapsedTo refers to original or decimated mesh
	OpenMesh::VPropHandleT<typename MeshT::VertexHandle> m_collapsedTo;				// References either the vertex in the original mesh we collapsed to or the vertex in the decimated mesh
};


} // namespace mufflon::renderer::decimaters::modules