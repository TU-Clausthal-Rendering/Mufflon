#pragma once

#include "util/int_types.hpp"
#include "util/filesystem.hpp"
#include <algorithm>
#include <chrono>
#include <ostream>
#include <string>
#include <string_view>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mufflon {

enum class ProfileLevel {
	LOW,
	HIGH,
	ALL
};

class CpuProfileState {
public:
	friend class CpuProfiler;

	using Microsecond = std::chrono::duration<u64, std::micro>;
	using WallClock = std::chrono::high_resolution_clock;
	using WallTimePoint = WallClock::time_point;

	class CpuProfileScope {
	public:
		CpuProfileScope(const CpuProfileScope&) = delete;
		CpuProfileScope(CpuProfileScope&&) = default;
		CpuProfileScope& operator=(const CpuProfileScope&) = delete;
		CpuProfileScope& operator=(CpuProfileScope&&) = delete;
		~CpuProfileScope();

	private:
		friend class CpuProfileState;

		CpuProfileScope(CpuProfileState& profiler);

		u64 m_startCycle;
		Microsecond m_startThreadTime;
		Microsecond m_startProcessTime;
		WallClock::time_point m_startWallTimepoint;
		CpuProfileState& m_profiler;
	};

	CpuProfileState(CpuProfileState*& active);

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

	// Starts a sample
	[[nodiscard]]
	CpuProfileScope start();
	// Resets the state and all sub-states
	void reset();
	// Saves the current sample data and resets only this profiler's state
	void create_snapshot();
	// Saves the current sample data and resets this and children's state
	void create_snapshot_all();
	std::ostream& save_snapshots(std::ostream& stream) const;
	std::ostream& save_current_state(std::ostream& stream) const;

private:
	// Sample data
	struct SampleData {
		u64 totalCpuCycles = 0u;
		Microsecond totalThreadTime;
		Microsecond totalProcessTime;
		Microsecond totalWallTime;
		u64 sampleCount = 0u;
	};

	// Add a new sample
	void add_sample(const CpuProfileScope& scope);

	SampleData m_currentSample;
	// Stores the parent profiler
	CpuProfileState* const m_parent;
	CpuProfileState*& m_activeRef;
	std::unordered_map<std::string_view, CpuProfileState> m_children;
	std::vector<SampleData> m_snapshots;
};

class CpuProfiler {
public:
	[[nodiscard]]
	std::optional<CpuProfileState::CpuProfileScope> start(std::string_view name, ProfileLevel level = ProfileLevel::HIGH);
	static CpuProfiler& instance();

	void reset();
	// Saves the current sample data and resets this and children's state
	void create_snapshot(std::string_view name);
	void create_snapshot_from(std::string_view name);
	void create_snapshot_all();
	void save_current_state(std::string_view path) const;
	void save_snapshots(std::string_view path) const;
	std::string save_current_state() const;
	std::string save_snapshots() const;

	// Sets the level of profiling
	constexpr ProfileLevel get_profile_level() const noexcept {
		return m_activation;
	}
	void set_profile_level(ProfileLevel level) noexcept {
		m_activation = level;
	}

	// Enables/disables profiling alltogether
	constexpr bool is_enabled() const noexcept {
		return m_enabled;
	}
	void set_enabled(bool enabled) noexcept {
		m_enabled = enabled;
	}

private:
	CpuProfiler() = default;

	// Holds the top-level profilers
	bool m_enabled = false;
	ProfileLevel m_activation = ProfileLevel::LOW;
	CpuProfileState* m_active = nullptr;
	std::unordered_map<std::string_view, CpuProfileState> m_profilers;
};

} // namespace mufflon