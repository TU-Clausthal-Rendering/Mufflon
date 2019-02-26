#include "sil_decimater.hpp"

namespace mufflon::renderer {

template < class MeshT >
ImportanceModule<MeshT>::ImportanceModule(MeshT &mesh) :
	Base(mesh, false) {}

template < class MeshT >
void ImportanceModule<MeshT>::initialize() {
	// TODO
}

template < class MeshT >
float ImportanceModule<MeshT>::collapse_priority(const CollapseInfo& ci) {
	// Gather in ring
	float importance = Base::mesh().property(m_importancePropHandle, ci.v0);
	for(auto ringVertexHandle = Base::mesh().vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
		float factor = 1.0f;
		if(*ringVertexHandle == ci.v1)
			factor += 0.001f;

		importance += Base::mesh().property(m_importancePropHandle, *ringVertexHandle) * factor;
	}
	return importance;
}

template < class MeshT >
void ImportanceModule<MeshT>::use_collapse_history(bool val) {
	m_useCollapseHistory = val;
}

template < class MeshT >
void ImportanceModule<MeshT>::set_importance_map(const ImportanceMap& importanceMap, const u32 meshIndex) {
	m_importanceMap = &importanceMap;
	m_meshIndex = meshIndex;
	m_importancePropHandle = importanceMap.get_importance_property(meshIndex);
}

// Post-process halfedge collapse (accumulate importance)
template < class MeshT >
void ImportanceModule<MeshT>::postprocess_collapse(const CollapseInfo& ci) {
	Base::mesh().property(m_importancePropHandle, ci.v1) += Base::mesh().property(m_importancePropHandle, ci.v0);
}

template class ImportanceModule<scene::geometry::PolygonMeshType>;

} // namespace mufflon::renderer