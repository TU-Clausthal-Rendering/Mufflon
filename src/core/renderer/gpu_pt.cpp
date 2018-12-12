#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"

namespace mufflon::renderer {

GpuPathTracer::~GpuPathTracer() {
	if(m_scenePtr != nullptr)
		cuda::check_error(cudaFree(m_scenePtr));
}

void GpuPathTracer::iterate(OutputHandler& outputBuffer) {
	// TODO: call sample in a parallel way for each output pixel
	// TODO: pass scene data to kernel!
	auto scope = Profiler::instance().start<GpuProfileState>("GPU PT iteration", ProfileLevel::LOW);

	if(m_reset) {
		// TODO: reset output buffer
	}

	this->iterate(outputBuffer.get_resolution(),
				  std::move(outputBuffer.begin_iteration<Device::CUDA>(m_reset)));
	m_reset = false;

	outputBuffer.end_iteration<Device::CUDA>();
	Profiler::instance().create_snapshot_all();
}

void GpuPathTracer::reset() {
	m_reset = true;
}

void GpuPathTracer::load_scene(scene::SceneHandle scene) {
	m_currentScene = scene;
	// Make sure the scene is loaded completely for the use on CPU side
	m_currentScene->synchronize<Device::CPU>();
	scene::SceneDescriptor<Device::CUDA> sceneDesc = m_currentScene->get_descriptor<Device::CUDA>({}, {}, {});
	if (m_scenePtr != nullptr)
		cuda::check_error(cudaFree(m_scenePtr));
	cuda::check_error(cudaMalloc(&m_scenePtr, sizeof(*m_scenePtr)));
	cuda::check_error(cudaMemcpy(m_scenePtr, &sceneDesc, sizeof(*m_scenePtr), cudaMemcpyDefault));
	m_reset = true;
}

} // namespace mufflon::renderer
