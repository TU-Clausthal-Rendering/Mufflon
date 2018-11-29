#pragma once

#include "profiling.hpp"
#include "util/int_types.hpp"
#include <algorithm>
#include <chrono>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mufflon {

class CpuProfileState : public ProfileState {
public:
	friend class Profiler;

	using Microsecond = std::chrono::duration<u64, std::micro>;
	using WallClock = std::chrono::high_resolution_clock;
	using WallTimePoint = WallClock::time_point;

	CpuProfileState(ProfileState*& active);

	// Abstractions from OS APIs
	static u64 get_cpu_cycle();
	static Microsecond get_thread_time();
	static Microsecond get_process_time();
	static WallTimePoint get_wall_timepoint();

	// CPU cylces
	constexpr u64 get_total_cpu_cycles() const noexcept {
		return m_currentSample.totalCpuCycles;
	}
	constexpr double get_average_cpu_cycles() const noexcept {
		return static_cast<double>(m_currentSample.totalCpuCycles) / static_cast<double>(std::max<u64>(1u, m_currentSample.sampleCount));
	}

	// Thread time
	constexpr Microsecond get_total_thread_time() const noexcept {
		return m_currentSample.totalThreadTime;
	}
	constexpr std::chrono::duration<double, std::micro> get_average_thread_time() const noexcept {
		return m_currentSample.totalThreadTime / static_cast<double>(std::max<u64>(1u, m_currentSample.sampleCount));
	}

	// Process time
	constexpr Microsecond get_total_process_time() const noexcept {
		return m_currentSample.totalProcessTime;
	}
	constexpr std::chrono::duration<double, std::micro> get_average_process_time() const noexcept {
		return m_currentSample.totalProcessTime / static_cast<double>(std::max<u64>(1u, m_currentSample.sampleCount));
	}

	// Wall time
	constexpr Microsecond get_total_wall_time() const noexcept {
		return m_currentSample.totalWallTime;
	}
	constexpr std::chrono::duration<double, std::micro> get_average_wall_time() const noexcept {
		return m_currentSample.totalWallTime / static_cast<double>(std::max<u64>(1u, m_currentSample.sampleCount));
	}

	// Total number of times we profiled
	constexpr u64 get_sample_count() const noexcept {
		return m_currentSample.sampleCount;
	}

	// Saves the current sample data and resets only this profiler's state
	virtual void create_snapshot() override;
	// Resets the state and all sub-states
	virtual void reset_sample() override;

	// Returns the memory used
	static std::size_t get_total_memory();
	static std::size_t get_free_memory();
	static std::size_t get_used_memory();

protected:
	virtual void start_sample() override;
	virtual void create_sample() override;
	// Save the created snapshots
	virtual std::ostream& save_profiler_snapshots(std::ostream& stream) const override;
	// Save the current profiler state
	virtual std::ostream& save_profiler_current_state(std::ostream& stream) const override;

private:
	// Sample data
	struct SampleData {
		u64 totalCpuCycles = 0u;
		Microsecond totalThreadTime;
		Microsecond totalProcessTime;
		Microsecond totalWallTime;
		u64 sampleCount = 0u;
	};

	// Time points from when start is called
	u64 m_startCpuCycle = 0u;
	Microsecond m_startThreadTime;
	Microsecond m_startProcessTime;
	WallClock::time_point m_startWallTimepoint;

	// Current sample data
	SampleData m_currentSample;
	std::vector<SampleData> m_snapshots;
};

} // namespace mufflon