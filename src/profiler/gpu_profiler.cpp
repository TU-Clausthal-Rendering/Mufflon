#include "gpu_profiler.hpp"
#include "util/log.hpp"
#include "core/cuda/error.hpp"
#include <cuda_runtime.h>

namespace mufflon {

GpuProfileState::GpuProfileState(ProfileState*& active) :
	ProfileState(active) {}

void GpuProfileState::create_sample() {
	cudaEvent_t endEvent;
	cuda::check_error(cudaEventCreateWithFlags(&endEvent, cudaEventBlockingSync));
	cuda::check_error(cudaEventSynchronize(endEvent));
	float ms = 0.f;
	cuda::check_error(cudaEventElapsedTime(&ms, m_startEvent, endEvent));
	m_currentSample.totalWallTime += std::chrono::duration_cast<Microsecond>(WallClock::now() - m_startWallTimepoint);
	m_currentSample.totalGpuTime = Microsecond(static_cast<u64>(ms * 1000.f));
	++m_currentSample.sampleCount;
}

void GpuProfileState::start_sample() {
	// Bump up the active profiler
	m_startWallTimepoint = WallClock::now();
	cuda::check_error(cudaEventCreate(&m_startEvent));
}

void GpuProfileState::reset_sample() {
	m_currentSample = SampleData{};
}

void GpuProfileState::create_snapshot() {
	m_snapshots.push_back(m_currentSample);
	m_currentSample = SampleData();
}

std::ostream& GpuProfileState::save_profiler_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << ",snapshots:" << m_snapshots.size() << ",type:gpu\n";
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalWallTime.count() << ','
			<< snapshot.totalGpuTime.count() << ","
			<< snapshot.sampleCount << '\n';
	}
	return stream;
}

std::ostream& GpuProfileState::save_profiler_current_state(std::ostream& stream) const {
	// Stores the current state as a CSV
	stream << ",type:gpu\n" << m_currentSample.totalWallTime.count() << ','
		<< m_currentSample.totalGpuTime.count() << ","
		<< m_currentSample.sampleCount << '\n';
	return stream;
}

std::size_t GpuProfileState::get_total_memory() {
	std::size_t free = 0u;
	std::size_t total = 0u;
	cuda::check_error(cudaMemGetInfo(&free, &total));
	return total;
}

std::size_t GpuProfileState::get_free_memory() {
	std::size_t free = 0u;
	std::size_t total = 0u;
	cuda::check_error(cudaMemGetInfo(&free, &total));
	return  free;
}

std::size_t GpuProfileState::get_used_memory() {
	return get_total_memory() - get_free_memory();
}

}