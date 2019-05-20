#pragma once

#include "lt_params.hpp"
#include "core/renderer/renderer_base.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuLightTracer final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuLightTracer();
	~CpuLightTracer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Lighttracer"; }
	StringView get_short_name() const noexcept final { return "LT"; }

	void on_reset() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	LtParameters m_params = {};
	std::vector<math::Rng> m_rngs;
};

} // namespace mufflon::renderer