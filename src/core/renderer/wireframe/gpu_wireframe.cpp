#include "gpu_wireframe.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"
#include <random>

namespace mufflon::renderer {

namespace gpuwireframe_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						WireframeTargets::RenderBufferType<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						const u32* seeds, const WireframeParameters& params);

} // namespace gpuwireframe_detail

GpuWireframe::GpuWireframe () :
	m_params{}
	//m_rng{ static_cast<u32>(std::random_device{}()) }
{}

void GpuWireframe::post_reset() {
	ResetEvent resetFlags { { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
								ResetEvent::ALL : get_reset_event() } };
	if(resetFlags.resolution_changed()) {
		m_seeds = std::make_unique<u32[]>(m_outputBuffer.get_num_pixels());
		m_seedsPtr = make_udevptr_array<Device::CUDA, u32>(m_outputBuffer.get_num_pixels());
	}
}

void GpuWireframe::iterate() {
	//auto scope = Profiler::instance().start<GpuProfileState>("GPU Wireframe iteration", ProfileLevel::LOW);

	for(int i = 0; i < m_outputBuffer.get_num_pixels(); ++i)
		m_seeds[i] = static_cast<u32>(m_rng.next());
	copy(m_seedsPtr.get(), m_seeds.get(), sizeof(u32) * m_outputBuffer.get_num_pixels());

	// TODO: pass scene data to kernel!
	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(m_outputBuffer.get_width() - 1) / blockDims.x,
		1u + static_cast<u32>(m_outputBuffer.get_height() - 1) / blockDims.y,
		1u
	};

	cuda::check_error(gpuwireframe_detail::call_kernel(gridDims, blockDims, std::move(m_outputBuffer),
													   m_sceneDesc.get(), m_seedsPtr.get(), m_params));
}

} // namespace mufflon::renderer
