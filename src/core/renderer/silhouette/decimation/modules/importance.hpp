#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::silhouette::decimation::modules {

// General importance decimater based on threshold
template < class MeshT = scene::geometry::PolygonMeshType >
class ImportanceDecimationModule : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(ImportanceDecimationModule, MeshT, ImportanceDecimationModule);

	ImportanceDecimationModule(MeshT& mesh) :
		Base(mesh, false) {}
	virtual ~ImportanceDecimationModule() = default;
	ImportanceDecimationModule(const ImportanceDecimationModule&) = delete;
	ImportanceDecimationModule(ImportanceDecimationModule&&) = delete;
	ImportanceDecimationModule& operator=(const ImportanceDecimationModule&) = delete;
	ImportanceDecimationModule& operator=(ImportanceDecimationModule&&) = delete;

	void set_properties(MeshT& originalMesh,
						OpenMesh::VPropHandleT<float> importanceDensity,
						const float threshold) {
		m_originalMesh = &originalMesh;
		m_importanceDensity = importanceDensity;
		m_threshold = threshold;
	}

	float collapse_priority(const CollapseInfo& ci) final {
		// We rely on the assumption that the original and the target mesh are identical
		// Gather in ring
		float importance = m_originalMesh->property(m_importanceDensity, ci.v0);

		for(auto circIter = Base::mesh().cvv_iter(ci.v0); circIter.is_valid(); ++circIter) {
			float factor = 1.0f;
			// TODO: what's up with that factor?
			if(*circIter == ci.v1)
				factor += 0.001f;

			importance += m_originalMesh->property(m_importanceDensity, *circIter) * factor;
		}

		if(importance < m_threshold)
			return Base::ILLEGAL_COLLAPSE;
		return importance;
	}

	void postprocess_collapse(const CollapseInfo& ci) final {
		// TODO: proper redistribution!
		m_originalMesh->property(m_importanceDensity, ci.v1) += m_originalMesh->property(m_importanceDensity, ci.v0);
	}

private:
	MeshT* m_originalMesh;

	// Original mesh properties
	OpenMesh::VPropHandleT<float> m_importanceDensity;								// Temporary storage to keep track of the remapped importance

	float m_threshold;
};

} // namespace mufflon::renderer::silhouette::decimation::modules