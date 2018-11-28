#include "cpu_profiler.hpp"
#include "util/log.hpp"
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#else // _WIN32
#include <pthread.h>
#include <time.h>
#include <ctime>
#endif // _WIN32

namespace mufflon {


CpuProfileState::CpuProfileScope::CpuProfileScope(CpuProfileState& profiler) :
	m_startCycle(get_cpu_cycle()),
	m_startThreadTime(get_thread_time()),
	m_startProcessTime(get_process_time()),
	m_startWallTimepoint(get_wall_timepoint()),
	m_profiler(profiler)
{
}

CpuProfileState::CpuProfileScope::~CpuProfileScope() {
	m_profiler.add_sample(*this);
}

CpuProfileState::CpuProfileState(CpuProfileState*& active) :
	m_activeRef(active),
	m_parent(active)
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

void CpuProfileState::add_sample(const CpuProfileScope& scope) {
	m_currentSample.totalCpuCycles += get_cpu_cycle() - scope.m_startCycle;
	m_currentSample.totalThreadTime += get_thread_time() - scope.m_startThreadTime;
	m_currentSample.totalProcessTime += get_process_time() - scope.m_startProcessTime;
	m_currentSample.totalWallTime += std::chrono::duration_cast<Microsecond>(WallClock::now() - scope.m_startWallTimepoint);
	++m_currentSample.sampleCount;

	// Change the active profiler back to whatever came before
	m_activeRef = m_parent;
}

CpuProfileState::CpuProfileScope CpuProfileState::start() {
	// Bump up the active profiler
	m_activeRef = this;
	return CpuProfileScope(*this);
}

void CpuProfileState::reset() {
	m_currentSample = SampleData{};
	for(auto& child : m_children)
		child.second.reset();
}

void CpuProfileState::create_snapshot() {
	m_snapshots.push_back(m_currentSample);
	m_currentSample = SampleData();
}

void CpuProfileState::create_snapshot_all() {
	m_snapshots.push_back(m_currentSample);
	for(auto& child : m_children)
		child.second.create_snapshot_all();
	m_currentSample = SampleData();
}

std::ostream& CpuProfileState::save_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << "children:" << m_children.size() << ",snapshots:" << m_snapshots.size() << '\n';
	for(const auto& snapshot : m_snapshots) {
		stream << snapshot.totalCpuCycles << ',' << snapshot.totalThreadTime.count() << ','
			<< snapshot.totalProcessTime.count() << ',' << snapshot.totalWallTime.count() << ','
			<< snapshot.sampleCount << '\n';
	}
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second.save_snapshots(stream);
	}
	return stream;
}

std::ostream& CpuProfileState::save_current_state(std::ostream& stream) const {
	// Stores the current state as a CSV
	stream << "children:" << m_children.size() << '\n';
	stream << m_currentSample.totalCpuCycles << ',' << m_currentSample.totalThreadTime.count() << ','
		<< m_currentSample.totalProcessTime.count() << ',' << m_currentSample.totalWallTime.count() << ','
		<< m_currentSample.sampleCount << '\n';
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second.save_current_state(stream);
	}
	return stream;
}

CpuProfiler& CpuProfiler::instance() {
	static CpuProfiler instance;
	return instance;
}

std::optional<CpuProfileState::CpuProfileScope> CpuProfiler::start(std::string_view name, ProfileLevel level) {
	if(m_enabled && static_cast<std::underlying_type_t<ProfileLevel>>(level)
					<= static_cast<std::underlying_type_t<ProfileLevel>>(m_activation)) {
		if(m_active != nullptr) {
			// Use cascaded profiler
			auto iter = m_active->m_children.find(name);
			if(iter == m_active->m_children.end())
				iter = m_active->m_children.emplace(name, CpuProfileState{ m_active }).first;
			return iter->second.start();
		} else {
			// Use top level profiler
			auto iter = m_profilers.find(name);
			if(iter == m_profilers.end())
				iter = m_profilers.emplace(name, CpuProfileState{ m_active }).first;
			return iter->second.start();
		}
	}

	return std::nullopt;
}

void CpuProfiler::reset() {
	for(auto& profiler : m_profilers)
		profiler.second.reset();
}

void CpuProfiler::create_snapshot(std::string_view name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		auto iter = m_active->m_children.find(name);
		if(iter != m_active->m_children.end())
			iter->second.create_snapshot();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second.create_snapshot();
	}
}

void CpuProfiler::create_snapshot_from(std::string_view name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		auto iter = m_active->m_children.find(name);
		if(iter != m_active->m_children.end())
			iter->second.create_snapshot_all();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second.create_snapshot_all();
	}
}

void CpuProfiler::create_snapshot_all() {
	for(auto& profiler : m_profilers)
		profiler.second.create_snapshot_all();
}

void CpuProfiler::save_current_state(std::string_view path) const {
	fs::path file = path;
	std::ofstream fileStream(file);
	if(fileStream.bad()) {
		logError("[CpuProfiler::save_current_state] could not open output file '",
				 file.string(), "'");
		return;
	}
	fileStream.exceptions(std::ios::failbit);

	try {
		for(auto& profiler : m_profilers) {
			fileStream << '"' << profiler.first << "\",";
			profiler.second.save_current_state(fileStream);
		}
	} catch(const std::exception& e) {
		logError("[CpuProfiler::save_current_state] Failed to save profiling results: ",
				 e.what());
	}
}

void CpuProfiler::save_snapshots(std::string_view path) const {
	fs::path file = path;
	std::ofstream fileStream(file);
	if(fileStream.bad()) {
		logError("[CpuProfiler::save_snapshots] could not open output file '",
				 file.string(), "'");
		return;
	}
	fileStream.exceptions(std::ios::failbit);

	try {
		for(auto& profiler : m_profilers) {
			fileStream << '"' << profiler.first << "\",";
			profiler.second.save_snapshots(fileStream);
		}
	} catch(const std::exception& e) {
		logError("[CpuProfiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
}

std::string CpuProfiler::save_current_state() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second.save_current_state(stream);
		}
	} catch(const std::exception& e) {
		logError("[CpuProfiler::save_current_state] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

std::string CpuProfiler::save_snapshots() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second.save_snapshots(stream);
		}
	} catch(const std::exception& e) {
		logError("[CpuProfiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

}