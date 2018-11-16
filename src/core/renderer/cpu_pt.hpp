#pragma once

#include "renderer.hpp"
#include "core/scene/scene.hpp"

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

	virtual void iterate(OutputHandler& outputBuffer) const override;
	virtual void reset() override;
private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer) const;

	scene::SceneHandle m_currentScene;
};

} // namespace mufflon::renderer