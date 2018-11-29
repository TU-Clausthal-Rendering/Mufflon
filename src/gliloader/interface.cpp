#include "plugin/texture_interface.h"
#include "util/log.hpp"
#include <gli/gl.hpp>
#include <gli/gli.hpp>
#include <mutex>

// Helper macros for error checking and logging
#define FUNCTION_NAME __func__
#define CHECK(x, name, retval)													\
	do {																		\
		if(!x) {																\
			logError("[", FUNCTION_NAME, "] Violated condition (" #name ")");	\
			return retval;														\
		}																		\
	} while(0)
#define CHECK_NULLPTR(x, name, retval)											\
	do {																		\
		if(x == nullptr) {														\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (nullptr)");		\
			return retval;														\
		}																		\
	} while(0)

using namespace mufflon;

namespace {

void(*s_logCallback)(const char*, int);

Boolean load_openexr(const char* path, TextureData* texData) {
	// TODO
	return false;
}

// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
}

bool get_format(gli::format format, TextureData& texData) {
	switch(format) {
		case gli::format::FORMAT_R8_UINT_PACK8: texData.format = TextureFormat::FORMAT_R8U; texData.sRgb = false; return true;
		case gli::format::FORMAT_R8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_R8U; texData.sRgb = true; return true;
		case gli::format::FORMAT_RG8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RG8U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RG8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RG8U; texData.sRgb = true; return true;
		case gli::format::FORMAT_RGB8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RGB8U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGB8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RGB8U; texData.sRgb = true; return true;
		case gli::format::FORMAT_RGBA8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGBA8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = true; return true;
		case gli::format::FORMAT_R16_UINT_PACK16: texData.format = TextureFormat::FORMAT_R16U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RG16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RG16U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGB16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RGB16U; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGBA16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RGBA16U; texData.sRgb = false; return true;
		case gli::format::FORMAT_R32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_R32F; texData.sRgb = false; return true;
		case gli::format::FORMAT_RG32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RG32F; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGB32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGB32F; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGBA32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA32F; texData.sRgb = false; return true;
		case gli::format::FORMAT_RGB9E5_UFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGB9E5; texData.sRgb = false; return true;
		default: return false;
	}
}

} // namespace

Boolean set_logger(void(*logCallback)(const char*, int)) {
	static bool initialized = false;
	s_logCallback = logCallback;
	if(!initialized) {
		registerMessageHandler(delegateLog);
		disableStdHandler();
		initialized = true;
	}
	return true;
}

Boolean can_load_texture_format(const char* ext) {
	return std::strncmp(ext, ".dds", 4u) == 0
		|| std::strncmp(ext, ".ktx", 4u) == 0
		|| std::strncmp(ext, ".exr", 4u) == 0;
}

Boolean load_texture(const char* path, TextureData* texData) {
	CHECK_NULLPTR(path, "texture path", false);
	CHECK_NULLPTR(path, "texture return data", false);

	// Check if we need to load OpenEXR
	std::string_view pathView = path;
	if(pathView.length() >= 4u && pathView.substr(pathView.length() - 4u).compare(".exr") == 0)
		return load_openexr(path, texData);

	gli::texture tex = gli::load(path);
	if(tex.empty()) {
		logError("[", FUNCTION_NAME, "] Failed to open texture file '", pathView, "'");
		return false;
	}
	
	// Determine the format
	if(!get_format(tex.format(), *texData)) {
		logError("[", FUNCTION_NAME, "] Unknown texture format of texture '", pathView, "'");
		return false;
	}
	if(tex.faces() != 1u) {
		logError("[", FUNCTION_NAME, "] Cannot load cubemaps yet: '", pathView, "'");
		return false;
	}
	if(tex.layers() != 0u) {
		logError("[", FUNCTION_NAME, "] Cannot load texture arrays yet: '", pathView, "'");
		return false;
	}
	// TODO: load layers (aka texture arrays)
	// TODO: load faces (aka cubemap support)

	texData->layers = 1u;
	auto data = tex.data(0u, 0u, 0u);
	std::size_t size = tex.size(0u);
	texData->data = new std::uint8_t[size];
	std::memcpy(texData->data, data, size);

	return false;
}
