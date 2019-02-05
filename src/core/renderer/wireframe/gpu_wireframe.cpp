#include "gpu_wireframe.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include <random>

namespace mufflon::renderer {

namespace gpuwireframe_detail {

void call_kernel(const dim3& gridDims, const dim3& blockDims,
				 RenderBuffer<Device::CUDA>&& outputBuffer,
				 scene::SceneDescriptor<Device::CUDA>* scene,
				 const u32* seeds, const WireframeParameters& params);

} // namespace gpuwireframe_detail

GpuWireframe::GpuWireframe () : m_params{} {}

GpuWireframe::~GpuWireframe () {
	if(m_scenePtr != nullptr)
		cuda::check_error(cudaFree(m_scenePtr));
}

void GpuWireframe::iterate(OutputHandler& outputBuffer) {
	// TODO: call sample in a parallel way for each output pixel
	// TODO: pass scene data to kernel!
	if(m_reset) {
		// TODO: reset output buffer
		// Reacquire scene descriptor (partially?)
		scene::SceneDescriptor<Device::CUDA> sceneDesc = m_currentScene->get_descriptor<Device::CUDA>({}, {}, {}, outputBuffer.get_resolution());
		cuda::check_error(cudaMemcpy(m_scenePtr, &sceneDesc, sizeof(*m_scenePtr), cudaMemcpyDefault));
	}

	this->iterate(outputBuffer.get_resolution(),
				  std::move(outputBuffer.begin_iteration<Device::CUDA>(m_reset)));
	m_reset = false;

	outputBuffer.end_iteration<Device::CUDA>();
}

void GpuWireframe::reset() {
	m_reset = true;
}

void GpuWireframe::load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) {
	if(scene != m_currentScene) {
		m_currentScene = scene;
		// Make sure the scene is loaded completely for the use on CPU side
		m_currentScene->synchronize<Device::CUDA>();
		scene::SceneDescriptor<Device::CUDA> sceneDesc = m_currentScene->get_descriptor<Device::CUDA>({}, {}, {}, resolution);
		if(m_scenePtr == nullptr)
			cuda::check_error(cudaMalloc(&m_scenePtr, sizeof(*m_scenePtr)));
		cuda::check_error(cudaMemcpy(m_scenePtr, &sceneDesc, sizeof(*m_scenePtr), cudaMemcpyDefault));
		m_reset = true;
	}
}

void GpuWireframe::iterate(Pixel imageDims,
							RenderBuffer<Device::CUDA> outputBuffer) const {

	std::unique_ptr<u32[]> rnds = std::make_unique<u32[]>(imageDims.x * imageDims.y);
	math::Rng rng{ static_cast<u32>(std::random_device()()) };
	for(int i = 0; i < imageDims.x*imageDims.y; ++i)
		rnds[i] = static_cast<u32>(rng.next());
	u32* devRnds = nullptr;
	cuda::check_error(cudaMalloc(&devRnds, sizeof(u32) * imageDims.x * imageDims.y));
	cuda::check_error(cudaMemcpy(devRnds, rnds.get(), sizeof(u32) * imageDims.x * imageDims.y,
					  cudaMemcpyDefault));

	// TODO: pass scene data to kernel!
	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(imageDims.x - 1) / blockDims.x,
		1u + static_cast<u32>(imageDims.y - 1) / blockDims.y,
		1u
	};

	// TODO
	cuda::check_error(cudaGetLastError());
	gpuwireframe_detail::call_kernel(gridDims, blockDims, std::move(outputBuffer),
									 m_scenePtr, devRnds, m_params);
	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(devRnds));
}

} // namespace mufflon::renderer
