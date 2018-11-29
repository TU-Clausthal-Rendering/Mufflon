#include "cpu_profiler.hpp"
#include "util/log.hpp"
#ifdef _WIN32
#include <windows.h>
#else // _WIN32
#include <pthread.h>
#include <time.h>
#include <ctime>
#endif // _WIN32

namespace mufflon {

CpuProfileState::CpuProfileState(ProfileState*& active) :
	ProfileState(active)
{}

u64 CpuProfileState::get_cpu_cycle() {
#ifdef _WIN32
	ULONG64 cycles = 0u;
	if(!QueryProcessCycleTime(GetCurrentProcess(), &cycles)) {
		logError("[CpuProfileState::get_cpu_cycle] Failed to optain CPU cycle count");
		return 0u;
	}
	return cycles;
#else // _WIN32
	return 0u;
#endif // _WIN32
}

CpuProfileState::Microsecond CpuProfileState::get_thread_time() {
#ifdef _WIN32
	FILETIME creationTime;
	FILETIME exitTime;
	FILETIME kernelTime{ 0u };
	FILETIME userTime{ 0u };
	if(!::GetThreadTimes(::GetCurrentThread(), &creationTime,
						&exitTime, &kernelTime, &userTime)) {
		logError("[CpuProfileState::get_process_time] Failed to optain process times");
		return Microsecond(0u);
	}
	return Microsecond((static_cast<u64>(kernelTime.dwHighDateTime) << 32u)
					   + static_cast<u64>(kernelTime.dwLowDateTime)
					   + (static_cast<u64>(userTime.dwHighDateTime) << 32u)
					   + static_cast<u64>(userTime.dwLowDateTime));
#else // _WIN32
	struct timespec currTime;
	clockid_t threadClockId;
	if(::pthread_getcpuclockid(::pthread_self(), &threadClockId) != 0) {
		logError("[CpuProfileState::get_process_time] Failed to optain thread clock ID");
		return Microsecond(0u);
	}
	if(::clock_gettime(threadClockId, &currTime) != 0) {
		logError("[CpuProfileState::get_process_time] Failed to optain thread time");
		return Microsecond(0u);
	}
	return Microsecond(static_cast<u64>((currTime.tv_sec * 1e9 + currTime.tv_nsec) / 1e3));
#endif // _WIN32
}

CpuProfileState::Microsecond CpuProfileState::get_process_time() {
#ifdef _WIN32
	FILETIME creationTime;
	FILETIME exitTime;
	FILETIME kernelTime{ 0u };
	FILETIME userTime{ 0u };
	if(!::GetProcessTimes(::GetCurrentProcess(), &creationTime,
						&exitTime, &kernelTime, &userTime)) {
		logError("[CpuProfileState::get_process_time] Failed to optain process times");
		return Microsecond(0u);
	}
	return Microsecond((static_cast<u64>(kernelTime.dwHighDateTime) << 32u)
		+ static_cast<u64>(kernelTime.dwLowDateTime)
		+ (static_cast<u64>(userTime.dwHighDateTime) << 32u)
		+ static_cast<u64>(userTime.dwLowDateTime));
#else // _WIN32
	// TODO: this is kinda bad
	return Microsecond(static_cast<u64>(static_cast<double>(std::clock()) / CLOCKS_PER_SEC));
#endif // _WIN32
}

CpuProfileState::WallTimePoint CpuProfileState::get_wall_timepoint() {
	return WallClock::now();
}

void CpuProfileState::create_sample() {
	m_currentSample.totalCpuCycles += get_cpu_cycle() - m_startCpuCycle;
	m_currentSample.totalThreadTime += get_thread_time() - m_startThreadTime;
	m_currentSample.totalProcessTime += get_process_time() - m_startProcessTime;
	m_currentSample.totalWallTime += std::chrono::duration_cast<Microsecond>(WallClock::now() - m_startWallTimepoint);
	++m_currentSample.sampleCount;
}

void CpuProfileState::start_sample() {
	// Bump up the active profiler
	m_startCpuCycle = get_cpu_cycle();
	m_startThreadTime = get_thread_time();
	m_startProcessTime = get_process_time();
	m_startWallTimepoint = get_wall_timepoint();
}

void CpuProfileState::reset_sample() {
	m_currentSample = SampleData{};
}

void CpuProfileState::create_snapshot() {
	m_snapshots.push_back(m_currentSample);
	m_currentSample = SampleData();
}

std::ostream& CpuProfileState::save_profiler_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << ",snapshots:" << m_snapshots.size() << '\n';
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalCpuCycles << ',' << snapshot.totalThreadTime.count() << ','
			<< snapshot.totalProcessTime.count() << ',' << snapshot.totalWallTime.count() << ','
			<< snapshot.sampleCount << '\n';
	}
	return stream;
}

std::ostream& CpuProfileState::save_profiler_current_state(std::ostream& stream) const {
	// Stores the current state as a CSV
	stream << m_currentSample.totalCpuCycles << ',' << m_currentSample.totalThreadTime.count() << ','
		<< m_currentSample.totalProcessTime.count() << ',' << m_currentSample.totalWallTime.count() << ','
		<< m_currentSample.sampleCount << '\n';
	return stream;
}

std::size_t CpuProfileState::get_memory_used() {
	// TODO
	return 0u;
}

std::size_t CpuProfileState::get_total_memory() {
	// TODO
	return 0u;
}

}