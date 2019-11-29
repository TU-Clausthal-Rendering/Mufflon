#include "cpu_profiler.hpp"
#include "util/log.hpp"
#ifdef _WIN32
#include <windows.h>
#include <Psapi.h>
#else // _WIN32
#include <pthread.h>
#include <time.h>
#include <ctime>
#include <unistd.h>
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
	return Microsecond(((static_cast<u64>(kernelTime.dwHighDateTime) << 32u)
					   + static_cast<u64>(kernelTime.dwLowDateTime)
					   + (static_cast<u64>(userTime.dwHighDateTime) << 32u)
					   + static_cast<u64>(userTime.dwLowDateTime)) / 10u);
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
	return Microsecond(((static_cast<u64>(kernelTime.dwHighDateTime) << 32u)
		+ static_cast<u64>(kernelTime.dwLowDateTime)
		+ (static_cast<u64>(userTime.dwHighDateTime) << 32u)
		+ static_cast<u64>(userTime.dwLowDateTime)) / 10u);
#else // _WIN32
	// TODO: this is kinda bad
	return Microsecond(static_cast<u64>(static_cast<double>(std::clock()) / CLOCKS_PER_SEC));
#endif // _WIN32
}

CpuProfileState::WallTimePoint CpuProfileState::get_wall_timepoint() {
	return WallClock::now();
}

void CpuProfileState::create_sample() {
	const u64 cpuCycles = get_cpu_cycle() - m_startCpuCycle;
	const Microsecond threadTime = get_thread_time() - m_startThreadTime;
	const Microsecond processTime = get_process_time() - m_startProcessTime;
	const Microsecond wallTime = std::chrono::duration_cast<Microsecond>(WallClock::now() - m_startWallTimepoint);
	m_currentSample.totalCpuCycles += cpuCycles;
	m_currentSample.totalThreadTime += threadTime;
	m_currentSample.totalProcessTime += processTime;
	m_currentSample.totalWallTime += wallTime;
	m_totalSample.totalCpuCycles += cpuCycles;
	m_totalSample.totalThreadTime += threadTime;
	m_totalSample.totalProcessTime += processTime;
	m_totalSample.totalWallTime += wallTime;
	++m_currentSample.sampleCount;
	++m_totalSample.sampleCount;
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
	stream << ",type:cpu,snapshots:" << m_snapshots.size() << '\n';
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalCpuCycles << ',' << snapshot.totalThreadTime.count() << ','
			<< snapshot.totalProcessTime.count() << ',' << snapshot.totalWallTime.count() << ','
			<< snapshot.sampleCount << '\n';
	}
	return stream;
}

std::ostream& CpuProfileState::save_profiler_total(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << ",type:cpu\n";
	stream << m_totalSample.totalCpuCycles << ',' << m_totalSample.totalThreadTime.count() << ','
		<< m_totalSample.totalProcessTime.count() << ',' << m_totalSample.totalWallTime.count() << ','
		<< m_totalSample.sampleCount << '\n';
	return stream;
}

std::ostream& CpuProfileState::save_profiler_current_state(std::ostream& stream) const {
	// Stores the current state as a CSV
	stream << ",type:cpu\n" << m_currentSample.totalCpuCycles << ',' << m_currentSample.totalThreadTime.count() << ','
		<< m_currentSample.totalProcessTime.count() << ',' << m_currentSample.totalWallTime.count() << ','
		<< m_currentSample.sampleCount << '\n';
	return stream;
}

std::ostream& CpuProfileState::save_profiler_total_and_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << ",type:cpu,currsnapshots:" << m_snapshots.size() << '\n';
	stream << m_totalSample.totalCpuCycles << ',' << m_totalSample.totalThreadTime.count() << ','
		<< m_totalSample.totalProcessTime.count() << ',' << m_totalSample.totalWallTime.count() << ','
		<< m_totalSample.sampleCount << '\n';
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalCpuCycles << ',' << snapshot.totalThreadTime.count() << ','
			<< snapshot.totalProcessTime.count() << ',' << snapshot.totalWallTime.count() << ','
			<< snapshot.sampleCount << '\n';
	}
	return stream;
}

std::size_t CpuProfileState::get_total_memory() {
#ifdef _WIN32
	ULONGLONG memKb;
	if(!::GetPhysicallyInstalledSystemMemory(&memKb)) {
		logError("[CpuProfileState::get_total_memory] Failed to optain physical memory size");
		return 0u;
}
	return memKb * 1024;
#else // _WIN32
	return sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
#endif // _WIN32
}

std::size_t CpuProfileState::get_free_memory() {
#ifdef _WIN32
	MEMORYSTATUSEX state;
	state.dwLength = sizeof(state);
	if(!::GlobalMemoryStatusEx(&state)) {
		logError("[CpuProfileState::get_free_memory] Failed to optain free memory");
		return 0u;
	}
	return state.ullAvailPhys;
#else // _WIN32
	// TODO: this isn't really accurate for an OS anymore...
	return get_total_memory() - get_used_memory();
#endif // _WIN32
}

std::size_t CpuProfileState::get_used_memory() {
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS counters;
	if(!::GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
		logError("[CpuProfileState::get_used_memory] Failed to optain process memory");
		return 0u;
	}
	return counters.WorkingSetSize;
#else // _WIN32
	long mem = 0;
	FILE* fp = nullptr;
	if((fp = std::fopen("/proc/self/statm", "r")) == nullptr) {
		logError("[CpuProfileState::get_used_memory] Failed to open stats file");
		return 0u;
	}
	if(std::fscanf(fp, "%*s%ld", &mem) != 1) {
		logError("[CpuProfileState::get_used_memory] Failed to find process memory");
		mem = 0u;
	}
	if(std::fclose(fp) != 0)
		logWarning("[CpuProfileState::get_used_memory] Failed to close stats file");
	return mem * sysconf(_SC_PAGE_SIZE);
#endif // _WIN32
}

}