#include "gpu_pt.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"
#include <random>

namespace mufflon::renderer {

namespace gpupt_detail {

cudaError_t call_kernel(RenderBuffer<Device::CUDA>&& outputBuffer,
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
	cuda::check_error(gpupt_detail::call_kernel(std::move(m_outputBuffer),
												m_sceneDesc.get(), m_rngs.get(), m_params));
}

void GpuPathTracer::on_reset() {
	gpupt_detail::init_rngs(m_outputBuffer.get_num_pixels(), m_rngs.get());
}

void GpuPathTracer::on_descriptor_requery() {
	if(!m_rngs || (m_rngs.get_deleter().get_size() != static_cast<std::size_t>(m_outputBuffer.get_num_pixels())))
		m_rngs = make_udevptr_array<Device::CUDA, math::Rng, false>(m_outputBuffer.get_num_pixels());
}

} // namespace mufflon::renderer
