#include "gpu_lt.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/world_container.hpp"
#include "profiler/gpu_profiler.hpp"
#include <random>

namespace mufflon::renderer {

namespace gpult_detail {

cudaError_t call_kernel(RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const LtParameters& params);

void init_rngs(u32 num, int seed, math::Rng* rngs);

} // namespace gpupt_detail

GpuLightTracer::GpuLightTracer() :
	m_params{}
	//m_rng{ static_cast<u32>(std::random_device{}()) }
{}

void GpuLightTracer::iterate() {
	//auto scope = Profiler::instance().start<GpuProfileState>("GPU LT iteration", ProfileLevel::LOW);

	cuda::check_error(gpult_detail::call_kernel(std::move(m_outputBuffer),
												m_sceneDesc.get(), m_rngs.get(), m_params));
}

void GpuLightTracer::on_reset() {
	int seed = m_params.seed * (m_outputBuffer.get_num_pixels() + 1);
	gpult_detail::init_rngs(m_outputBuffer.get_num_pixels(), seed, m_rngs.get());
}

void GpuLightTracer::post_descriptor_requery() {
	if(!m_rngs || (m_rngs.get_deleter().get_size() != static_cast<std::size_t>(m_outputBuffer.get_num_pixels())))
		m_rngs = make_udevptr_array<Device::CUDA, math::Rng, false>(m_outputBuffer.get_num_pixels());
}

} // namespace mufflon::renderer
