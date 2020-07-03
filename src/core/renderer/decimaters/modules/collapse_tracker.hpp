#pragma once

#include "util/int_types.hpp"
#include "core/scene/geometry/polygon_mesh.hpp"
#include "core/renderer/decimaters/util/collapse_history.hpp"
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

	void set_properties(CollapseHistory* history, const u32 frameIndex) {
		m_history = history;
		m_frameIndex = frameIndex;
	}

	void postprocess_collapse(const CollapseInfo& ci) final {
		auto history = m_history[ci.v0.idx()];

		// Check if we constrained this collapse (meaning we won't update the frame)
		if(!history.collapsed || history.collapsedTo != static_cast<u32>(ci.v1.idx())) {
			history.frameIndex = m_frameIndex;
		}
		history.collapsedTo = static_cast<u32>(ci.v1.idx());
		history.collapsed = true;
		m_history[ci.v0.idx()] = history;
	}

private:
	CollapseHistory* m_history;
	u32 m_frameIndex;
};

} // namespace mufflon::renderer::decimaters::modules