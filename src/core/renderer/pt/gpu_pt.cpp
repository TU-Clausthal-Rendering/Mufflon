#include "gpu_pt.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"
#include <random>

namespace mufflon::renderer {

namespace gpupt_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const PtParameters& params);

void init_rngs(u32 num, math::Rng* rngs);

} // namespace gpupt_detail

GpuPathTracer::GpuPathTracer() :
	m_params{}
	//m_rng{ static_cast<u32>(std::random_device{}()) }
{}

void GpuPathTracer::iterate() {
	//auto scope = Profiler::instance().start<GpuProfileState>("GPU PT iteration", ProfileLevel::LOW);

	copy(&m_sceneDesc->lightTree.posGuide, &m_params.neeUsePositionGuide, sizeof(bool));
	 
	// TODO: pass scene data to kernel!
	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(m_outputBuffer.get_width() - 1) / blockDims.x,
		1u + static_cast<u32>(m_outputBuffer.get_height() - 1) / blockDims.y,
		1u
	};

	cuda::check_error(gpupt_detail::call_kernel(gridDims, blockDims, std::move(m_outputBuffer),
												m_sceneDesc.get(), m_rngs.get(), m_params));

	//Profiler::instance().create_snapshot_all();
}

void GpuPathTracer::on_reset() {
	gpupt_detail::init_rngs(m_outputBuffer.get_num_pixels(), m_rngs.get());
}

void GpuPathTracer::post_descriptor_requery() {
	m_rngs = make_udevptr_array<Device::CUDA, math::Rng>(m_outputBuffer.get_num_pixels());
}

} // namespace mufflon::renderer
