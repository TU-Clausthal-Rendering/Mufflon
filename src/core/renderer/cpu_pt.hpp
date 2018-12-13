#pragma once

#include "renderer.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuPathTracer : public IRenderer {
public:
	// Initialize all resources required by this renderer.
	CpuPathTracer();
	~CpuPathTracer() = default;

	virtual void iterate(OutputHandler& outputBuffer) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; }
	virtual bool has_scene() const noexcept override { return m_currentScene != nullptr; }
	virtual void load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) override;
private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
				const scene::SceneDescriptor<Device::CPU>& scene);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	bool m_reset = true;
	ParameterHandler<PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide> m_params = {};
	scene::SceneHandle m_currentScene = nullptr;
	std::vector<math::Xoroshiro128> m_rngs;
	scene::SceneDescriptor<Device::CPU> m_sceneDesc;
};

} // namespace mufflon::renderer