#pragma once

#include "pt_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/targets/render_targets.hpp"
#include <future>
#include <vector>

namespace mufflon::cameras {
struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

class HybridPathTracer final : public IRenderer {
public:
	using RenderBufferCpu = typename PtTargets::template RenderBufferType<Device::CPU>;
	using RenderBufferCuda = typename PtTargets::template RenderBufferType<Device::CUDA>;
	using OutputHandlerType = typename PtTargets::OutputHandlerType;

	// Initialize all resources required by this renderer.
	HybridPathTracer(mufflon::scene::WorldContainer& world);
	~HybridPathTracer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Pathtracer"; }
	static constexpr StringView get_short_name_static() noexcept { return "PT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	bool uses_device(Device device) const noexcept override { return may_use_device(device); }
	static constexpr bool may_use_device(Device device) noexcept {
		return Device::CPU == device || Device::CUDA == device;
	}

	bool pre_iteration(IOutputHandler& outputBuffer) override;
	void post_iteration(IOutputHandler& outputBuffer) override;

	void post_reset() final;

	std::unique_ptr<IOutputHandler> create_output_handler(int width, int height) override {
		return std::make_unique<OutputHandlerType>(width, height);
	}

private:
	void iterate_cpu();
	void iterate_cuda(const std::chrono::high_resolution_clock::time_point& begin,
					  std::promise<std::chrono::high_resolution_clock::duration>&& duration);

	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	PtParameters m_params = {};
	int m_currYSplit;
	int m_nextYSplit;
	std::vector<math::Rng> m_rngsCpu;
	unique_device_ptr<Device::CUDA, math::Rng[]> m_rngsCuda;

	RenderBufferCpu m_outputBufferCpu;
	RenderBufferCuda m_outputBufferCuda;

	scene::SceneDescriptor<Device::CPU> m_sceneDescCpu;
	unique_device_ptr<Device::CUDA, scene::SceneDescriptor<Device::CUDA>> m_sceneDescCuda;
};

} // namespace mufflon::renderer