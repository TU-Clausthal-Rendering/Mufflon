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

class CpuBidirPathTracer final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuBidirPathTracer();
	~CpuBidirPathTracer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Bidirectional Pathtracer"; }
	StringView get_short_name() const noexcept final { return "BPT"; }

	void on_reset() final;

private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	bool m_reset = true;
	BptParameters m_params = {};
	scene::SceneHandle m_currentScene = nullptr;
	std::vector<math::Rng> m_rngs;
	scene::SceneDescriptor<Device::CPU> m_sceneDesc;
};

} // namespace mufflon::renderer