#include "mff_interface.h"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include "profiler/cpu_profiler.hpp"
#include "mffloader/parsing/json_loader.hpp"
#include "mffloader/parsing/binary.hpp"
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
#define CHECK_NULLPTR(x, name, retval)											\
	do {																		\
		if((x) == nullptr) {													\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (nullptr)");		\
			return retval;														\
		}																		\
	} while(0)
#define CATCH_ALL(retval)														\
	} catch(const std::exception& e) {											\
		logError("[", FUNCTION_NAME, "] Exception caught: ", e.what());			\
		s_lastError = e.what();													\
		return retval;															\
	}

using namespace mufflon;
using namespace mff_loader;

struct MufflonLoaderInstance {
	MufflonInstanceHdl mffInst;
	std::atomic<json::JsonLoader*> jsonLoader = nullptr;
	json::FileVersion fileVersion;
	fs::path binPath{};
};

namespace {

std::string s_lastError;

} // namespace

Boolean loader_set_log_level(LogLevel level) {
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

MufflonLoaderInstanceHdl loader_initialize(MufflonInstanceHdl mffInstHdl) {
	return new MufflonLoaderInstance{ mffInstHdl, nullptr, {}, fs::path{} };
}

void loader_destroy(MufflonLoaderInstanceHdl mffLoaderInstHdl) {
	if(mffLoaderInstHdl != nullptr)
		delete mffLoaderInstHdl;
}

LoaderStatus loader_load_json(MufflonLoaderInstanceHdl hdl, const char* path) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", LoaderStatus::LOADER_ERROR);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	const auto filePath = fs::u8path(path);

	// Perform some error checking
	if (!fs::exists(filePath)) {
		logError("[", FUNCTION_NAME, "] File '", fs::canonical(filePath).u8string(), "' does not exist");
		return LoaderStatus::LOADER_ERROR;
	}
	if (fs::is_directory(filePath)) {
		logError("[", FUNCTION_NAME, "] Path '", fs::canonical(filePath).u8string(), "' is a directory, not a file");
		return LoaderStatus::LOADER_ERROR;
	}
	if (filePath.extension() != ".json")
		logWarning("[", FUNCTION_NAME, "] Scene file does not end with '.json'; attempting to parse it anyway");

	try {
		// Clear the world
		world_clear_all(mffLoaderInst.mffInst);
		json::JsonLoader loader{ mffLoaderInst.mffInst, filePath };
		mufflon_set_lod_loader(mffLoaderInst.mffInst, loader_load_lod, loader_load_object_material_indices, hdl);
		mffLoaderInst.jsonLoader.store(&loader);
		if(!loader.load_file(mffLoaderInst.binPath, &mffLoaderInst.fileVersion))
			return LoaderStatus::LOADER_ABORT;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] ", e.what());
		mffLoaderInst.jsonLoader.store(nullptr);
		return LoaderStatus::LOADER_ERROR;
	}
	mffLoaderInst.jsonLoader.store(nullptr);
	return LoaderStatus::LOADER_SUCCESS;
	CATCH_ALL(LoaderStatus::LOADER_ERROR)
}

LoaderStatus loader_save_scene(MufflonLoaderInstanceHdl hdl, const char* path) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", LoaderStatus::LOADER_ERROR);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	auto filePath = fs::u8path(path);

	// Perform some error checking
	if (fs::exists(filePath)) {
		filePath.replace_extension(".mff");
		if (fs::is_directory(filePath)) {
			logError("[", FUNCTION_NAME, "] Path '", filePath.u8string(), "' is already a directory");
			return LoaderStatus::LOADER_ERROR;
		}
		filePath.replace_extension(".json");
		if (fs::is_directory(filePath)) {
			logError("[", FUNCTION_NAME, "] Path '", filePath.u8string(), "' is already a directory");
			return LoaderStatus::LOADER_ERROR;
		}
	}
	try {
	exprt::SceneExporter exporter{ mffLoaderInst.mffInst, filePath, mffLoaderInst.binPath };
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

Boolean loader_load_lod(MufflonLoaderInstanceHdl hdl, ObjectHdl obj, u32 lod, Boolean asReduced) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", false);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	if(obj == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid object handle");
		return false;
	}
	u32 objId;
	if(!object_get_id(obj, &objId))
		return false;
	std::string status;
	binary::BinaryLoader loader{ mffLoaderInst.mffInst, status };
	loader.load_lod(mffLoaderInst.binPath, obj, objId, lod, asReduced != 0u);
	return true;
	CATCH_ALL(false)
}

Boolean loader_load_object_material_indices(MufflonLoaderInstanceHdl hdl, const uint32_t objId,
											uint16_t* indexBuffer, uint32_t* readIndices) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", false);
	CHECK_NULLPTR(indexBuffer, "material index buffer", false);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	if(mffLoaderInst.fileVersion < json::JsonLoader::PER_OBJECT_UNIQUE_MAT_INDICES) {
		if(readIndices != nullptr)
			*readIndices = 0u;
		return true;
	}
	std::string status;
	binary::BinaryLoader loader{ mffLoaderInst.mffInst, status };
	const auto count = loader.read_unique_object_material_indices(mffLoaderInst.binPath, objId, indexBuffer);
	if(readIndices != nullptr)
		*readIndices = count;
	return true;
	CATCH_ALL(false)
}

Boolean loader_abort(MufflonLoaderInstanceHdl hdl) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", false);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	if(json::JsonLoader* loader = mffLoaderInst.jsonLoader.load(); loader != nullptr) {
		loader->abort_load();
		return true;
	}
	logError("[", FUNCTION_NAME, "] No loader running that could be aborted");
	return false;
	CATCH_ALL(false)
}

const char* loader_get_loading_status(MufflonLoaderInstanceHdl hdl) {
	TRY
	CHECK_NULLPTR(hdl, "loader instance handle", nullptr);
	auto& mffLoaderInst = *static_cast<MufflonLoaderInstance*>(hdl);
	if(json::JsonLoader* loader = mffLoaderInst.jsonLoader.load(); loader != nullptr)
		return loader->get_loading_stage().c_str();
	return "";
	CATCH_ALL(nullptr)
}

void loader_profiling_enable() {
	TRY
	Profiler::loader().set_enabled(true);
	CATCH_ALL(;)
}

void loader_profiling_disable() {
	TRY
	Profiler::loader().set_enabled(false);
	CATCH_ALL(;)
}

Boolean loader_profiling_set_level(ProfilingLevel level) {
	TRY
	switch(level) {
		case ProfilingLevel::PROFILING_OFF:
			Profiler::loader().set_enabled(false);
			return true;
		case ProfilingLevel::PROFILING_LOW:
			Profiler::loader().set_profile_level(ProfileLevel::LOW);
			return true;
		case ProfilingLevel::PROFILING_HIGH:
			Profiler::loader().set_profile_level(ProfileLevel::HIGH);
			return true;
		case ProfilingLevel::PROFILING_ALL:
			Profiler::loader().set_profile_level(ProfileLevel::ALL);
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
	Profiler::loader().save_current_state(path);
	return true;
	CATCH_ALL(false)
}

Boolean loader_profiling_save_snapshots(const char* path) {
	TRY
	if(path == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid file path (nullptr)");
		return false;
	}
	Profiler::loader().save_snapshots(path);
	return true;
	CATCH_ALL(false)
}

Boolean loader_profiling_save_total_and_snapshots(const char* path) {
	TRY
	if(path == nullptr) {
		logError("[", FUNCTION_NAME, "] Invalid file path (nullptr)");
		return false;
	}
	Profiler::loader().save_total_and_snapshots(path);
	return true;
	CATCH_ALL(false)
}

const char* loader_profiling_get_current_state() {
	TRY
		static thread_local std::string str;
	str = Profiler::loader().save_current_state();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* loader_profiling_get_snapshots() {
	TRY
	static thread_local std::string str;
	str = Profiler::loader().save_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* loader_profiling_get_total() {
	TRY
	static thread_local std::string str;
	str = Profiler::loader().save_total();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* loader_profiling_get_total_and_snapshots() {
	TRY
	static thread_local std::string str;
	str = Profiler::loader().save_total_and_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

void loader_profiling_reset() {
	TRY
	Profiler::loader().reset_all();
	CATCH_ALL(;)
}
