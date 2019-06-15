#pragma once

#include "pt_params.hpp"
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

class CpuPathTracer final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuPathTracer();
	~CpuPathTracer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Pathtracer"; }
	static constexpr StringView get_short_name_static() noexcept { return "PT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	PtParameters m_params = {};
	std::vector<math::Rng> m_rngs;
};

} // namespace mufflon::renderer