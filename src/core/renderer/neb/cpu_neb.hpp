#pragma once

#include "neb_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/photon_map.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

template < typename ExtensionT >
class PathVertex;
namespace { using NebPathVertex = PathVertex<struct NebVertexExt>; }

class CpuNextEventBacktracking final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuNextEventBacktracking();
	~CpuNextEventBacktracking() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Next Event Backtracking"; }
	StringView get_short_name() const noexcept final { return "NEB"; }

	void on_reset() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	NebParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	HashGridManager<NebPathVertex> m_viewVertexMapManager;
	HashGrid<Device::CPU, NebPathVertex> m_viewVertexMap;
};

} // namespace mufflon::renderer