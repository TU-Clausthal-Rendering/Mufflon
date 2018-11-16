#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/scene/scene.hpp"

namespace mufflon::renderer {

GpuPathTracer::GpuPathTracer(scene::SceneHandle scene) :
	m_currentScene(scene) {
	// Make sure the scene is loaded completely for the use on CPU side
	scene->synchronize<Device::CPU>();

	// The PT does not need additional memory resources like photon maps.
}

void GpuPathTracer::iterate(OutputHandler& outputBuffer) const {
	// TODO: call sample in a parallel way for each output pixel

	const ei::IVec2& resolution = m_currentScene->get_resolution();
	if(resolution.x <= 0 || resolution.y <= 0) {
		logError("[GpuPathTracer::iterate] Invalid resolution (<= 0)");
		return;
	}

	// TODO: pass scene data to kernel!
	this->iterate(m_currentScene->get_resolution(), std::move(m_currentScene->get_light_tree<Device::CUDA>()),
				  std::move(outputBuffer.begin_iteration<Device::CUDA>(false)));

	outputBuffer.end_iteration<Device::CUDA>();
}

void GpuPathTracer::reset() {
	// TODO
}

} // namespace mufflon::renderer