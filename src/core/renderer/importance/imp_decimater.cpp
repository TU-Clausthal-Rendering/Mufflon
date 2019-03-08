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

template < class MeshT >
MaxNormalDeviation<MeshT>::MaxNormalDeviation(MeshT &mesh) : Base(mesh, true) {
	this->set_max_deviation(90.f);
}

template < class MeshT >
float MaxNormalDeviation<MeshT>::collapse_priority(const CollapseInfo& ci) {
	// Compute the face normals before the collapse
	m_normalStorage.clear();
	for(typename Mesh::ConstVertexFaceIter iter(Base::mesh(), ci.v0); iter.is_valid(); ++iter) {
		typename Mesh::FaceHandle fh = *iter;
		if(fh != ci.fl && fh != ci.fr) {
			m_normalStorage.push_back(Base::mesh().calc_face_normal(fh));
		}
	}

	// simulate collapse
	Base::mesh().set_point(ci.v0, ci.p1);

	// check for flipping normals
	typename Mesh::Scalar c(1.0);
	u32 index = 0u;
	for(typename Mesh::ConstVertexFaceIter iter(Base::mesh(), ci.v0); iter.is_valid(); ++iter, ++index) {
		typename Mesh::FaceHandle fh = *iter;
		if(fh != ci.fl && fh != ci.fr) {
			typename const Mesh::Normal& n1 = m_normalStorage[index];
			typename Mesh::Normal n2 = Base::mesh().calc_face_normal(fh);

			c = dot(n1, n2);

			if(c < m_minCos)
				break;
		}
	}

	// undo simulation changes
	Base::mesh().set_point(ci.v0, ci.p0);

	return static_cast<float>((c < m_minCos) ? Base::ILLEGAL_COLLAPSE : Base::LEGAL_COLLAPSE);
}

template < class MeshT >
void MaxNormalDeviation<MeshT>::set_max_deviation(const double deviation) {
	m_maxDeviation = deviation;
	m_minCos = std::cos(m_maxDeviation);
}

template < class MeshT >
void MaxNormalDeviation<MeshT>::set_error_tolerance_factor(const double factor) {
	if(factor >= 0.0 && factor <= 1.0) {
		// the smaller the factor, the smaller max_deviation_ gets
		// thus creating a stricter constraint
		// division by error_tolerance_factor_ is for normalization
		double maxDeviation = (m_maxDeviation * 180.0 / ei::PI) * factor / this->error_tolerance_factor_;
		set_max_deviation(maxDeviation);
		this->error_tolerance_factor_ = factor;
	}
}

template < class MeshT >
double MaxNormalDeviation<MeshT>::get_max_devation() const noexcept {
	return m_maxDeviation;
}

template class ModImportance<scene::geometry::PolygonMeshType>;
template class MaxNormalDeviation<scene::geometry::PolygonMeshType>;

} // namespace mufflon::renderer::importance