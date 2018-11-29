#include "interface.h"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"
#include "loader/parsing/json_loader.hpp"
#include <stdexcept>
#include <mutex>
#include <iostream>

#define FUNCTION_NAME __func__

using namespace mufflon;
using namespace loader;

namespace {

void(*s_logCallback)(const char*, int);

// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
}

} // namespace

Boolean loader_set_logger(void(*logCallback)(const char*, int)) {
	static bool initialized = false;
	s_logCallback = logCallback;
	if(!initialized) {
		registerMessageHandler(delegateLog);
		disableStdHandler();
		initialized = true;
	}
	return true;
}

Boolean loader_load_json(const char* path) {
	fs::path filePath(path);

	// Perform some error checking
	if (!fs::exists(filePath)) {
		logError("[", FUNCTION_NAME, "] File '", fs::canonical(filePath).string(), "' does not exist");
		return false;
	}
	if (fs::is_directory(filePath)) {
		logError("[", FUNCTION_NAME, "] Path '", fs::canonical(filePath).string(), "' is a directory, not a file");
		return false;
	}
	if (filePath.extension() != ".json")
		logWarning("[", FUNCTION_NAME, "] Scene file does not end with '.json'; attempting to parse it anyway");

	try {
		json::JsonLoader loader{ filePath };
		loader.load_file();
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] ", e.what());
		return false;
	}

	std::cout << Profiler::instance().save_current_state() << std::endl;

	return true;
}