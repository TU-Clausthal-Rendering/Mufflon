#pragma once

#include "profiling.hpp"
#include "util/int_types.hpp"
#include <algorithm>
#include <chrono>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mufflon {

class GpuProfileState : public ProfileState {
public:
	friend class Profiler;

	using Microsecond = std::chrono::duration<u64, std::micro>;
	using WallClock = std::chrono::high_resolution_clock;
	using WallTimePoint = WallClock::time_point;

	GpuProfileState(ProfileState*& active);

	// Abstractions from OS APIs
	static WallTimePoint get_wall_timepoint();

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
		Microsecond totalWallTime;
		u64 sampleCount = 0u;
	};

	// Time points from when start is called
	WallClock::time_point m_startWallTimepoint;

	// Current sample data
	SampleData m_currentSample;
	std::vector<SampleData> m_snapshots;
};

} // namespace mufflon