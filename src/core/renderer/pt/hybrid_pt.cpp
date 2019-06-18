#pragma once

#include "hybrid_pt.hpp"
#include "pt_common.hpp"
#include "profiler/cpu_profiler.hpp"
#include "profiler/gpu_profiler.hpp"
#include "util/parallel.hpp"

namespace mufflon::renderer {

namespace hybridpt_detail {

cudaError_t call_kernel(RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const int yOffset,
						const PtParameters& params);

void init_rngs(u32 num, int seed, math::Rng* rngs);

} // namespace hybridpt_detail

HybridPathTracer::HybridPathTracer() {
	m_sceneDescCuda = make_udevptr<Device::CUDA, mufflon::scene::SceneDescriptor<Device::CUDA>>();
}


void HybridPathTracer::iterate() {
	m_currYSplit = static_cast<int>(std::clamp(m_params.split, 0.f, 1.f) * m_outputBufferCpu.get_height());

	// TODO: load balancing
	iterate_cuda();
	iterate_cpu();

	cudaDeviceSynchronize();
	//cuda::check_error(cudaGetLastError());
}

void HybridPathTracer::iterate_cpu() {
	auto scope = Profiler::instance().start<CpuProfileState>("Hybrid PT CPU iteration", ProfileLevel::HIGH);

	PtParameters ptParams;
	ptParams.seed = m_params.seed;
	ptParams.minPathLength = m_params.minPathLength;
	ptParams.maxPathLength = m_params.maxPathLength;
	ptParams.neeCount = m_params.neeCount;
	ptParams.neeUsePositionGuide = m_params.neeUsePositionGuide;

	m_sceneDescCpu.lightTree.posGuide = m_params.neeUsePositionGuide;

	// TODO: better pixel order?
	// TODO: different scheduling?

	const int PIXELS = m_outputBufferCpu.get_width() * m_currYSplit;
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < PIXELS; ++pixel) {
		Pixel coord{ pixel % m_outputBufferCpu.get_width(), pixel / m_outputBufferCpu.get_width() };
		pt_sample(m_outputBufferCpu, m_sceneDescCpu, ptParams, coord, m_rngsCpu[pixel]);
	}
}

void HybridPathTracer::iterate_cuda() {
	auto scope = Profiler::instance().start<CpuProfileState>("Hybrid PT CUDA iteration", ProfileLevel::HIGH);
	PtParameters ptParams;
	ptParams.seed = m_params.seed;
	ptParams.minPathLength = m_params.minPathLength;
	ptParams.maxPathLength = m_params.maxPathLength;
	ptParams.neeCount = m_params.neeCount;
	ptParams.neeUsePositionGuide = m_params.neeUsePositionGuide;

	//auto scope = Profiler::instance().start<GpuProfileState>("GPU PT iteration", ProfileLevel::LOW);
	copy(&m_sceneDescCuda->lightTree.posGuide, &m_params.neeUsePositionGuide, sizeof(bool));
	cuda::check_error(hybridpt_detail::call_kernel(std::move(m_outputBufferCuda),
												   m_sceneDescCuda.get(), m_rngsCuda.get(),
												   m_currYSplit, ptParams));
}

void HybridPathTracer::post_reset() {
	if(!m_rngsCuda || m_rngsCuda.get_deleter().get_size() != static_cast<std::size_t>(m_outputBufferCuda.get_num_pixels()))
		m_rngsCuda = make_udevptr_array<Device::CUDA, math::Rng, false>(m_outputBufferCuda.get_num_pixels());
	int seed = m_params.seed * (m_outputBufferCuda.get_num_pixels() + 1);
	hybridpt_detail::init_rngs(m_outputBufferCuda.get_num_pixels(), seed, m_rngsCuda.get());
	this->init_rngs(m_outputBufferCpu.get_num_pixels());
}

void HybridPathTracer::init_rngs(int num) {
	m_rngsCpu.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngsCpu[i] = math::Rng(i + seed);
}

bool HybridPathTracer::pre_iteration(OutputHandler& outputBuffer) {
	const bool needsReset = get_reset_event() != ResetEvent::NONE;
	auto[rb1, rb2] = outputBuffer.begin_iteration_hybrid<Device::CPU, Device::CUDA>(needsReset);
	m_outputBufferCpu = std::move(rb1);
	m_outputBufferCuda = std::move(rb2);
	m_currentIteration = outputBuffer.get_current_iteration();
	if(needsReset) {
		this->pre_reset();
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		auto desc = m_currentScene->get_descriptor<Device::CUDA>({}, {}, {});
		copy(m_sceneDescCuda.get(), &desc, sizeof(desc));
		m_sceneDescCpu = m_currentScene->get_descriptor<Device::CPU>({}, {}, {});
		this->clear_reset();
		return true;
	}
	return false;
}

void HybridPathTracer::post_iteration(OutputHandler& outputBuffer) {
	outputBuffer.sync_back<Device::CUDA, Device::CPU>(m_currYSplit);
	outputBuffer.end_iteration<Device::CPU>();
}

} // namespace mufflon::renderer