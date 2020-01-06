#include "hybrid_pt.hpp"
#include "pt_common.hpp"
#include "profiler/cpu_profiler.hpp"
#include "profiler/gpu_profiler.hpp"
#include "util/parallel.hpp"
#include <thread>

namespace mufflon::renderer {

namespace hybridpt_detail {

cudaError_t call_kernel(HybridPathTracer::RenderBufferCuda&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const int yOffset,
						const PtParameters& params);

void init_rngs(u32 num, int seed, math::Rng* rngs);

} // namespace hybridpt_detail

HybridPathTracer::HybridPathTracer(mufflon::scene::WorldContainer& world) :
	IRenderer{ world }
{
	m_sceneDescCuda = make_udevptr<Device::CUDA, mufflon::scene::SceneDescriptor<Device::CUDA>>();
}


void HybridPathTracer::iterate() {
	// Measure the time each device takes for load balancing
	std::promise<std::chrono::high_resolution_clock::duration> cudaPromise;
	auto cudaFuture = cudaPromise.get_future();
	const auto begin = std::chrono::high_resolution_clock::now();
	std::thread cudaLaunch(&HybridPathTracer::iterate_cuda, this, begin, std::move(cudaPromise));
	iterate_cpu();
	const auto cpuDuration = std::chrono::high_resolution_clock::now() - begin;
	cudaLaunch.join();
	const auto cudaDuration = cudaFuture.get();

	const auto cpuTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuDuration).count();
	const auto cudaTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(cudaDuration).count();
	// Compute the new screen split from the ratio of the times (including a dampening factor)
	const auto fraction = static_cast<float>(cpuTimeMs) / static_cast<float>(cudaTimeMs);
	// Dampening is based on the total time taken and uses a sigmoid (lower execution times are more
	// volatile due to fluctuations in scheduler etc.)
	const auto weight = 2 * (1.f / (1.f + std::exp(-2.f * (cpuTimeMs + cudaTimeMs) / 1000.f)) - 0.5f);
	m_nextYSplit = std::min(m_outputBufferCpu.get_height() - 1,
							std::max(0, m_currYSplit + static_cast<int>((1.f - fraction) * weight * m_outputBufferCpu.get_height())));
}

void HybridPathTracer::iterate_cpu() {
	//auto scope = Profiler::core().start<CpuProfileState>("Hybrid PT CPU iteration", ProfileLevel::HIGH);

	m_sceneDescCpu.lightTree.posGuide = m_params.neeUsePositionGuide;

	// TODO: better pixel order?
	// TODO: different scheduling?

	const int PIXELS = m_outputBufferCpu.get_width() * m_currYSplit;
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < PIXELS; ++pixel) {
		Pixel coord{ pixel % m_outputBufferCpu.get_width(), pixel / m_outputBufferCpu.get_width() };
		pt_sample(m_outputBufferCpu, m_sceneDescCpu, m_params, coord, m_rngsCpu[pixel]);
	}
}

void HybridPathTracer::iterate_cuda(const std::chrono::high_resolution_clock::time_point& begin,
									std::promise<std::chrono::high_resolution_clock::duration>&& duration) {
	//auto scope = Profiler::core().start<CpuProfileState>("Hybrid PT CUDA iteration", ProfileLevel::HIGH);

	//auto scope = Profiler::core().start<GpuProfileState>("GPU PT iteration", ProfileLevel::LOW);
	copy(&m_sceneDescCuda->lightTree.posGuide, &m_params.neeUsePositionGuide, sizeof(bool));
	cuda::check_error(hybridpt_detail::call_kernel(std::move(m_outputBufferCuda),
												   m_sceneDescCuda.get(), m_rngsCuda.get(),
												   m_currYSplit, m_params));
	cudaDeviceSynchronize();
	cuda::check_error(cudaGetLastError());
	duration.set_value(std::chrono::high_resolution_clock::now() - begin);
}

void HybridPathTracer::post_reset() {
	if(!m_rngsCuda || m_rngsCuda.get_deleter().get_size() != static_cast<std::size_t>(m_outputBufferCuda.get_num_pixels()))
		m_rngsCuda = make_udevptr_array<Device::CUDA, math::Rng, false>(m_outputBufferCuda.get_num_pixels());
	int seed = m_params.seed * (m_outputBufferCuda.get_num_pixels() + 1);
	hybridpt_detail::init_rngs(m_outputBufferCuda.get_num_pixels(), seed, m_rngsCuda.get());
	this->init_rngs(m_outputBufferCpu.get_num_pixels());

	// Set the initial load balancing guess to 50/50
	if(get_reset_event().resolution_changed())
		m_currYSplit = m_outputBufferCpu.get_height() / 2;
}

void HybridPathTracer::init_rngs(int num) {
	m_rngsCpu.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngsCpu[i] = math::Rng(i + seed);
}

bool HybridPathTracer::pre_iteration(IOutputHandler& outputBuffer) {
	const bool needsReset = get_reset_event() != ResetEvent::NONE;
	auto& out = dynamic_cast<OutputHandlerType&>(outputBuffer);
	auto[rb1, rb2] = out.template begin_iteration_hybrid<Device::CPU, Device::CUDA>(needsReset);
	m_outputBufferCpu = std::move(rb1);
	m_outputBufferCuda = std::move(rb2);
	m_currentIteration = out.get_current_iteration();
	if(needsReset) {
		this->pre_reset();
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		auto desc = m_currentScene->template get_descriptor<Device::CUDA>({}, {}, {});
		copy(m_sceneDescCuda.get(), &desc, sizeof(desc));
		m_sceneDescCpu = m_currentScene->template get_descriptor<Device::CPU>({}, {}, {});
		this->clear_reset();
		return true;
	}
	return false;
}

void HybridPathTracer::post_iteration(IOutputHandler& outputBuffer) {
	auto& out = dynamic_cast<OutputHandlerType&>(outputBuffer);
	out.template sync_back<Device::CUDA, Device::CPU>(m_currYSplit);
	m_currYSplit = m_nextYSplit;
	out.template end_iteration<Device::CPU>();
}

} // namespace mufflon::renderer