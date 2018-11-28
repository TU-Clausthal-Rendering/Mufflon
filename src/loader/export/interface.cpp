#include "interface.h"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"
#include "loader/parsing/json_loader.hpp"
#include <stdexcept>
#include <iostream>

#define FUNCTION_NAME __func__

using namespace mufflon;
using namespace loader;

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
	CpuProfiler::instance().create_snapshot_all();
	std::cout << CpuProfiler::instance().save_snapshots() << std::endl;

	return true;
}