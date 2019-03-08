#include "sil_decimater.hpp"

namespace mufflon::renderer::silhouette {

template < class MeshT >
ImportanceModule<MeshT>::ImportanceModule(MeshT &mesh) :
	Base(mesh, false) {}

template < class MeshT >
void ImportanceModule<MeshT>::initialize() {
	// TODO
}

template < class MeshT >
float ImportanceModule<MeshT>::collapse_priority(const CollapseInfo& ci) {
	const auto& propHandle = m_importanceMap->get_importance_property(m_meshIndex);
	float importance = Base::mesh().property(propHandle, ci.v0);
	u32 count = 0u;
	for(auto ringVertexHandle = Base::mesh().vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
		importance += Base::mesh().property(propHandle, *ringVertexHandle);
		++count;
	}
	importance /= static_cast<float>(count);
	if(importance > m_threshold)
		return -1.f;
	return importance;
}

template < class MeshT >
void ImportanceModule<MeshT>::use_collapse_history(bool val) {
	m_useCollapseHistory = val;
}

template < class MeshT >
void ImportanceModule<MeshT>::set_importance_map(ImportanceMap& importanceMap, const u32 meshIndex, const float threshold) {
	m_importanceMap = &importanceMap;
	m_meshIndex = meshIndex;
	m_threshold = threshold;
}

// Post-process halfedge collapse (accumulate importance)
template < class MeshT >
void ImportanceModule<MeshT>::postprocess_collapse(const CollapseInfo& ci) {
	//Base::mesh().property(m_importancePropHandle, ci.v1) += Base::mesh().property(m_importancePropHandle, ci.v0);
	m_importanceMap->collapse(m_meshIndex, ci.v0.idx(), ci.v1.idx());
}

template class ImportanceModule<scene::geometry::PolygonMeshType>;

} // namespace mufflon::renderer::silhouette