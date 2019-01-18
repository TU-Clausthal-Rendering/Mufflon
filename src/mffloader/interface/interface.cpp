#include "interface.h"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"
#include "mffloader/parsing/json_loader.hpp"
#include <atomic>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include "mffloader/export/scene_exporter.hpp"
#ifdef _WIN32
#include <combaseapi.h>
#endif // _WIN32

// Undefine windows API defines (bs...)
#undef ERROR

#define FUNCTION_NAME __func__

#define TRY try {
#define CATCH_ALL(retval)														\
	} catch(const std::exception& e) {											\
		logError("[", FUNCTION_NAME, "] Exception caught: ", e.what());			\
		s_lastError = e.what();													\
		return retval;															\
	}

using namespace mufflon;
using namespace mff_loader;

namespace {

std::string s_lastError;
std::atomic<json::JsonLoader*> s_jsonLoader = nullptr;

void(*s_logCallback)(const char*, int);

// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	TRY
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
	CATCH_ALL(;)
}

} // namespace

const char* loader_get_dll_error() {
	TRY
#ifdef _WIN32
		// For C# interop
		char* buffer = reinterpret_cast<char*>(::CoTaskMemAlloc(s_lastError.size() + 1u));
#else // _WIN32
		char* buffer = new char[s_lastError.size() + 1u];
#endif // _WIN32
	if(buffer == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to allocate state buffer");
		return nullptr;
	}
	std::memcpy(buffer, s_lastError.c_str(), s_lastError.size());
	buffer[s_lastError.size()] = '\0';
	return buffer;
	CATCH_ALL(nullptr)
}

bool loader_set_log_level(LogLevel level) {
	switch(level) {
		case LogLevel::LOG_PEDANTIC:
			mufflon::s_logLevel = LogSeverity::PEDANTIC;
			return true;
		case LogLevel::LOG_INFO:
			mufflon::s_logLevel = LogSeverity::INFO;
			return true;
		case LogLevel::LOG_WARNING:
			mufflon::s_logLevel = LogSeverity::WARNING;
			return true;
		case LogLevel::LOG_ERROR:
			mufflon::s_logLevel = LogSeverity::ERROR;
			return true;
		case LogLevel::LOG_FATAL_ERROR:
			mufflon::s_logLevel = LogSeverity::FATAL_ERROR;
			return true;
		default:
			logError("[", FUNCTION_NAME, "] Invalid log level");
			return false;
	}
}

Boolean loader_set_logger(void(*logCallback)(const char*, int)) {
	TRY
	static bool initialized = false;
	s_logCallback = logCallback;
	if(!initialized) {
		registerMessageHandler(delegateLog);
		disableStdHandler();
		initialized = true;
	}
	return true;
	CATCH_ALL(false)
}

LoaderStatus loader_load_json(const char* path) {
	TRY
	fs::path filePath(path);

	// Perform some error checking
	if (!fs::exists(filePath)) {
		logError("[", FUNCTION_NAME, "] File '", fs::canonical(filePath).string(), "' does not exist");
		return LoaderStatus::LOADER_ERROR;
	}
	if (fs::is_directory(filePath)) {
		logError("[", FUNCTION_NAME, "] Path '", fs::canonical(filePath).string(), "' is a directory, not a file");
		return LoaderStatus::LOADER_ERROR;
	}
	if (filePath.extension() != ".json")
		logWarning("[", FUNCTION_NAME, "] Scene file does not end with '.json'; attempting to parse it anyway");

	try {
		// Clear the world
		world_clear_all();
		json::JsonLoader loader{ filePath };
		s_jsonLoader.store(&loader);
		if(!loader.load_file())
			return LoaderStatus::LOADER_ABORT;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] ", e.what());
		return LoaderStatus::LOADER_ERROR;
	}

	return LoaderStatus::LOADER_SUCCESS;
	CATCH_ALL(LoaderStatus::LOADER_ERROR)
}

LoaderStatus loader_save_scene(const char* path) {
	TRY
	fs::path filePath(path);

	// Perform some error checking
	if (fs::exists(filePath)) {
		filePath.replace_extension(".mff");
		if (fs::is_directory(filePath)) {
			logError("[", FUNCTION_NAME, "] Path '", filePath.string(), "' is already a directory");
			return LoaderStatus::LOADER_ERROR;
		}
		filePath.replace_extension(".json");
		if (fs::is_directory(filePath)) {
			logError("[", FUNCTION_NAME, "] Path '", filePath.string(), "' is already a directory");
			return LoaderStatus::LOADER_ERROR;
		}
	}
	try {
	exprt::SceneExporter exporter{ filePath };
		if (!exporter.save_scene())
			return LoaderStatus::LOADER_ERROR;
	}
	catch (const std::exception& e) {
		logError("[", FUNCTION_NAME, "] ", e.what());
		return LoaderStatus::LOADER_ERROR;
	}

	return LoaderStatus::LOADER_SUCCESS;
	CATCH_ALL(LoaderStatus::LOADER_ERROR)
}


Boolean loader_abort() {
	if(json::JsonLoader* loader = s_jsonLoader.load(); loader != nullptr) {
		loader->abort_load();
		return true;
	}
	logError("[", FUNCTION_NAME, "] No loader running that could be aborted");
	return false;
}

void loader_profiling_enable() {
	TRY
	Profiler::instance().set_enabled(true);
	CATCH_ALL(;)
}

void loader_profiling_disable() {
	TRY
	Profiler::instance().set_enabled(false);
	CATCH_ALL(;)
}

Boolean loader_profiling_set_level(ProfilingLevel level) {
	TRY
	switch(level) {
		case ProfilingLevel::PROFILING_OFF:
			Profiler::instance().set_enabled(false);
			return true;
		case ProfilingLevel::PROFILING_LOW:
			Profiler::instance().set_profile_level(ProfileLevel::LOW);
			return true;
		case ProfilingLevel::PROFILING_HIGH:
			Profiler::instance().set_profile_level(ProfileLevel::HIGH);
			return true;
		case ProfilingLevel::PROFILING_ALL:
			Profiler::instance().set_profile_level(ProfileLevel::ALL);
			return true;
		default:
			logError("[", FUNCTION_NAME, "] invalid profiling level");
	}
	return false;
	CATCH_ALL(false)
}

Boolean loader_profiling_save_current_state(const char* path) {
	TRY
	if(path == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid file path (nullptr)");
		return false;
	}
	Profiler::instance().save_current_state(path);
	return true;
	CATCH_ALL(false)
}

Boolean loader_profiling_save_snapshots(const char* path) {
	TRY
	if(path == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid file path (nullptr)");
		return false;
	}
	Profiler::instance().save_snapshots(path);
	return true;
	CATCH_ALL(false)
}

Boolean loader_profiling_save_total_and_snapshots(const char* path) {
	TRY
	if(path == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid file path (nullptr)");
		return false;
	}
	Profiler::instance().save_total_and_snapshots(path);
	return true;
	CATCH_ALL(false)
}

const char* loader_profiling_get_current_state() {
	TRY
	std::string str = Profiler::instance().save_current_state();
#ifdef _WIN32
	// For C# interop
	char* buffer = reinterpret_cast<char*>(::CoTaskMemAlloc(str.size() + 1u));
#else // _WIN32
	char* buffer = new char[str.size() + 1u];
#endif // _WIN32
	if(buffer == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to allocate state buffer");
		return nullptr;
	}
	std::memcpy(buffer, str.c_str(), str.size());
	buffer[str.size()] = '\0';
	return buffer;
	CATCH_ALL(nullptr)
}

const char* loader_profiling_get_snapshots() {
	TRY
	std::string str = Profiler::instance().save_snapshots();
#ifdef _WIN32
	// For C# interop
	char* buffer = reinterpret_cast<char*>(::CoTaskMemAlloc(str.size() + 1u));
#else // _WIN32
	char* buffer = new char[str.size() + 1u];
#endif // _WIN32
	if(buffer == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to allocate state buffer");
		return nullptr;
	}
	std::memcpy(buffer, str.c_str(), str.size());
	buffer[str.size()] = '\0';
	return buffer;
	CATCH_ALL(nullptr)
}

const char* loader_profiling_get_total_and_snapshots() {
	TRY
	std::string str = Profiler::instance().save_total_and_snapshots();
#ifdef _WIN32
	// For C# interop
	char* buffer = reinterpret_cast<char*>(::CoTaskMemAlloc(str.size() + 1u));
#else // _WIN32
	char* buffer = new char[str.size() + 1u];
#endif // _WIN32
	if(buffer == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to allocate state buffer");
		return nullptr;
	}
	std::memcpy(buffer, str.c_str(), str.size());
	buffer[str.size()] = '\0';
	return buffer;
	CATCH_ALL(nullptr)
}

void loader_profiling_reset() {
	TRY
	Profiler::instance().reset_all();
	CATCH_ALL(;)
}