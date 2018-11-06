#pragma once

#include "core/scene/scene.hpp"

namespace mufflon::renderer {

class OutputHandler; // TODO: implement an output handler for various configurations (variance, guides, ...)

class CpuPathTracer {
public:
	// Initialize all resources required by this renderer.
	CpuPathTracer(scene::SceneHandle scene);

	void iterate(OutputHandler* outputBuffer) const;

	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, OutputHandler* outputBuffer) const;
private:
	scene::SceneHandle m_currentScene;
	std::vector<u8> m_cameraParams;
};

} // namespace mufflon::renderer