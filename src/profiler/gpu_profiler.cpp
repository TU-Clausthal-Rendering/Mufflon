#include "gpu_profiler.hpp"
#include "util/log.hpp"
#include "core/cuda/error.hpp"
#include <cuda_runtime.h>

namespace mufflon {

GpuProfileState::GpuProfileState(ProfileState*& active) :
	ProfileState(active) {}

GpuProfileState::WallTimePoint GpuProfileState::get_wall_timepoint() {
	return WallClock::now();
}

void GpuProfileState::create_sample() {
	m_currentSample.totalWallTime += std::chrono::duration_cast<Microsecond>(WallClock::now() - m_startWallTimepoint);
	++m_currentSample.sampleCount;
}

void GpuProfileState::start_sample() {
	// Bump up the active profiler
	m_startWallTimepoint = get_wall_timepoint();
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
	stream << ",snapshots:" << m_snapshots.size() << '\n';
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalWallTime.count() << ',' << snapshot.sampleCount << '\n';
	}
	return stream;
}

std::ostream& GpuProfileState::save_profiler_current_state(std::ostream& stream) const {
	// Stores the current state as a CSV
	stream << m_currentSample.totalWallTime.count() << ',' << m_currentSample.sampleCount << '\n';
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