#include "imp_decimater.hpp"
#include <ei/elementarytypes.hpp>
#include <cmath>

namespace mufflon::renderer::importance {

template < class MeshT >
ModImportance<MeshT>::ModImportance(MeshT &mesh) :
	Base(mesh, false)
{}

template < class MeshT >
void ModImportance<MeshT>::initialize() {
	// 1:1 mapping of importance
#if 0
	mAssert(m_importanceMapCreationMesh && m_importanceMapCreationMesh->n_vertices() == m_importanceMap->size() && Base::mesh().n_vertices() == m_meshDataCollapseHistory->size());

	// Region growing propagation
	if constexpr(USE_COLLAPSE_HISTORY) {
		// Propagate importance through collapse-history.
		/*for(auto vertexHandle : Base::mesh().vertices()) {
			if(Base::mesh().property(importancePropHandle, vertexHandle) < 0) {
				// Trace back until a valid importance is found.
				std::vector<unsigned int> importancePropagation;
				unsigned int collapsedTo = vertexHandle.idx();
				float goalImp = -1.0f;
				do {
					importancePropagation.push_back(collapsedTo);
					assert(collapsedTo != meshDataCollapseHistory->operator[](collapsedTo) && "Vertices can not collapse in loops!");
					collapsedTo = meshDataCollapseHistory->operator[](collapsedTo);
					goalImp = Base::mesh().property(importancePropHandle, Base::mesh().vertex_handle(collapsedTo));
				} while(goalImp < 0.0f);

				// Propagate found importance value.
				for(int i = (int)importancePropagation.size() - 1; i >= 0; --i)//unsigned int vertexIdx : importancePropagation)
				{
					goalImp *= 0.99f; // NEW - TODO - FACTOR?!!!
					Base::mesh().property(importancePropHandle, Base::mesh().vertex_handle(importancePropagation[i])) = goalImp;
				}
			}
		}*/
	} else {
		/*bool foundInvalid = false;
		do {
			foundInvalid = false;
			for(auto vertexHandle : Base::mesh().vertices()) {
				float importance = Base::mesh().property(m_importancePropHandle, vertexHandle);
				if(importance < 0) {
					foundInvalid = true;
				} else {
					for(auto vertexRing = Base::mesh().vv_iter(vertexHandle); vertexRing.is_valid(); ++vertexRing) {
						if(Base::mesh().property(m_importancePropHandle, *vertexRing) < 0)
							Base::mesh().property(m_importancePropHandle, *vertexRing) = importance;
					}
				}
			}
		} while(foundInvalid);*/
	}
	// Security check.
	/*for(auto vertexHandle : Base::mesh().vertices())
	{
		if(Base::mesh().property(importancePropHandle, vertexHandle) < 0)
		{
			std::cerr << "found illegal importance value!\n";
		}
	}*/
#endif
}

template < class MeshT >
float ModImportance<MeshT>::collapse_priority(const CollapseInfo& ci) {
	const auto& impPropHandle = m_importanceMap->get_importance_property(m_meshIndex);
	// Gather in ring
	float importance = Base::mesh().property(impPropHandle, ci.v0);
	for(auto ringVertexHandle = Base::mesh().vv_iter(ci.v0); ringVertexHandle.is_valid(); ++ringVertexHandle) {
		float factor = 1.0f;
		if(*ringVertexHandle == ci.v1)
			factor += 0.001f;

		importance += Base::mesh().property(impPropHandle, *ringVertexHandle) * factor;
	}
	return importance;
}

template < class MeshT >
void ModImportance<MeshT>::use_collapse_history(bool val) {
	m_useCollapseHistory = val;
}

template < class MeshT >
void ModImportance<MeshT>::set_importance_map(ImportanceMap& importanceMap, const u32 meshIndex) {
	m_importanceMap = &importanceMap;
	m_meshIndex = meshIndex;
}


	// Post-process halfedge collapse (accumulate importance)
template < class MeshT >
void ModImportance<MeshT>::postprocess_collapse(const CollapseInfo& ci) {
	m_importanceMap->collapse(m_meshIndex, ci.v0.idx(), ci.v1.idx());
}

template class ModImportance<scene::geometry::PolygonMeshType>;

} // namespace mufflon::renderer::importance