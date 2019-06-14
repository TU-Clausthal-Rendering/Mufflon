#pragma once

#include "bpt_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

template < typename ExtensionT >
class PathVertex;
namespace { using BptPathVertex = PathVertex<struct BptVertexExt>; }

class CpuBidirPathTracer final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuBidirPathTracer();
	~CpuBidirPathTracer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Bidirectional Pathtracer"; }
	static constexpr StringView get_short_name_static() noexcept { return "BPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

private:
	// Create one sample path (actual BPT algorithm)
	void sample(const Pixel coord, int pixel, RenderBuffer<Device::CPU>& outputBuffer,
				std::vector<BptPathVertex>& path);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	BptParameters m_params = {};
	std::vector<math::Rng> m_rngs;
};

} // namespace mufflon::renderer