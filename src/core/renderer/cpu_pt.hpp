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
	CpuPathTracer(scene::SceneHandle scene);

	virtual void iterate(OutputHandler& outputBuffer) override;
	virtual void reset() override;
private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void initRngs(int num);

	bool m_reset = true;
	scene::SceneHandle m_currentScene;
	std::vector<math::Xoroshiro128> m_rngs;
};

} // namespace mufflon::renderer