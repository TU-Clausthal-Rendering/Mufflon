#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::decimaters::silhouette::modules {

// Disallows collapse when a vertex has been marked as silhouette
template < class MeshT = scene::geometry::PolygonMeshType >
class SilhouetteDecimationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(SilhouetteDecimationModule, MeshT, SilhouetteDecimationModule);

	SilhouetteDecimationModule(MeshT& mesh) :
		Base(mesh, true) {}
	virtual ~SilhouetteDecimationModule() = default;
	SilhouetteDecimationModule(const SilhouetteDecimationModule&) = delete;
	SilhouetteDecimationModule(SilhouetteDecimationModule&&) = delete;
	SilhouetteDecimationModule& operator=(const SilhouetteDecimationModule&) = delete;
	SilhouetteDecimationModule& operator=(SilhouetteDecimationModule&&) = delete;

	void set_properties(MeshT& originalMesh, OpenMesh::VPropHandleT<bool> silhouette) {
		m_originalMesh = &originalMesh;
		m_silhouette = silhouette;
	}

	float collapse_priority(const CollapseInfo& ci) final {
		// Check if we're trying to collapse away or onto a silhouette vertex
		if(m_originalMesh->property(m_silhouette, ci.v0) || m_originalMesh->property(m_silhouette, ci.v1))
			return Base::ILLEGAL_COLLAPSE;
		return Base::LEGAL_COLLAPSE;
	}

private:
	MeshT* m_originalMesh;
	OpenMesh::VPropHandleT<bool> m_silhouette;
};

} // namespace mufflon::renderer::decimaters::silhouette::modules