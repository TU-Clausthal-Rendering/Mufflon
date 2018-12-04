#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"

namespace mufflon::renderer {

GpuPathTracer::GpuPathTracer(scene::SceneHandle scene) :
	m_currentScene(scene) {
	// Make sure the scene is loaded completely for the use on CPU side
	scene->synchronize<Device::CUDA>();

	// The PT does not need additional memory resources like photon maps.
}

void GpuPathTracer::iterate(OutputHandler& outputBuffer) {
	// TODO: call sample in a parallel way for each output pixel
	// TODO: pass scene data to kernel!
	auto scope = Profiler::instance().start<GpuProfileState>("GPU PT iteration", ProfileLevel::LOW);
	this->iterate(outputBuffer.get_resolution(),
				  std::move(outputBuffer.begin_iteration<Device::CUDA>(m_reset)));
	m_reset = false;

	outputBuffer.end_iteration<Device::CUDA>();
	Profiler::instance().create_snapshot_all();
}

void GpuPathTracer::reset() {
	m_reset = true;
}

} // namespace mufflon::renderer