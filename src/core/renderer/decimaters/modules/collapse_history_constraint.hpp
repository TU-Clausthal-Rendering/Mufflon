#pragma once

#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/renderer/decimaters/util/collapse_history.hpp"
#include "core/renderer/decimaters/octree/float_octree.inl"
#include "core/renderer/decimaters/octree/octree_manager.hpp"
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

namespace mufflon::renderer::decimaters::modules {

// Tracks the collapses of a mesh
template < class MeshT = scene::geometry::PolygonMeshType >
class CollapseHistoryConstraint : public OpenMesh::Decimater::ModBaseT<MeshT> {
public:
	DECIMATING_MODULE(CollapseHistoryConstraint, MeshT, CollapseHistoryConstraint);

	CollapseHistoryConstraint(MeshT& mesh) :
		Base(mesh, true) {}
	virtual ~CollapseHistoryConstraint() = default;
	CollapseHistoryConstraint(const CollapseHistoryConstraint&) = delete;
	CollapseHistoryConstraint(CollapseHistoryConstraint&&) = delete;
	CollapseHistoryConstraint& operator=(const CollapseHistoryConstraint&) = delete;
	CollapseHistoryConstraint& operator=(CollapseHistoryConstraint&&) = delete;

	void set_properties(const OctreeManager<FloatOctree>* importances,
						const CollapseHistory* history,
						const std::size_t objectIndex,
						const u32 currentFrame,
						const float factorThreshold) {
		m_importances = importances;
		m_history = history;
		m_objectIndex = objectIndex;
		m_currentFrame = currentFrame;
		m_factorThreshold = factorThreshold;
		m_deniedCount = 0u;
		m_allowedCount = 0u;
	}

	float collapse_priority(const CollapseInfo& ci) final {
		if(m_history[ci.v0.idx()].collapsed) {
			// We had a collapse; check if the target is still in the one-ring
			const auto oldTarget = Base::mesh().vertex_handle(m_history[ci.v0.idx()].collapsedTo);
			if(!oldTarget.is_valid())
				return Base::LEGAL_COLLAPSE;
			if(const auto heh = Base::mesh().find_halfedge(ci.v0, oldTarget);
			   heh.is_valid()) {// && !Base::mesh().status(heh).deleted()) {
				const auto point = util::pun<ei::Vec3>(Base::mesh().point(ci.v0));
				//const auto normal = util::pun<ei::Vec3>(Base::mesh().normal(ci.v0));
				const auto oldImp = m_importances[m_history[ci.v0.idx()].frameIndex][m_objectIndex].get_samples(point);
				//if(oldImp > 5000.f) {
					const auto newImp = m_importances[m_currentFrame][m_objectIndex].get_samples(point);
					const auto ratio = newImp / oldImp;
					if(ratio >= (1.f / m_factorThreshold) && ratio <= m_factorThreshold) {
						++m_deniedCount;
						return Base::ILLEGAL_COLLAPSE;
					}
				//}

				++m_allowedCount;
			}
		}
		return Base::LEGAL_COLLAPSE;
	}

	std::size_t get_denied_count() const noexcept {
		return m_deniedCount;
	}
	std::size_t get_allowed_count() const noexcept {
		return m_allowedCount;
	}

	/*void postprocess_collapse(const CollapseInfo& ci) final {
		// Assumes that the two meshes are initially equal
		m_history[ci.v0.idx()].collapsedTo = static_cast<u32>(ci.v1.idx());
		m_history[ci.v0.idx()].collapsed = true;
	}*/

private:
	const OctreeManager<FloatOctree>* m_importances;
	const CollapseHistory* m_history;
	std::size_t m_objectIndex;
	u32 m_currentFrame;
	float m_factorThreshold;
	std::size_t m_deniedCount;
	std::size_t m_allowedCount;
};

} // namespace mufflon::renderer::decimaters::modules