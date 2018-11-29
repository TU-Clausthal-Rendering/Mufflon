#include "plugin/texture_interface.h"
#include "util/log.hpp"
#include <stbi/stb_image.h>
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

TextureFormat get_int_format(int components) {
	switch(components) {
		case 1: return TextureFormat::FORMAT_R32F;
		case 2: return TextureFormat::FORMAT_RG32F;
		case 3: return TextureFormat::FORMAT_RGB32F;
		case 4: return TextureFormat::FORMAT_RGBA32F;
		default: return TextureFormat::FORMAT_NUM;
	}
}

TextureFormat get_float_format(int components) {
	switch(components) {
		case 1: return TextureFormat::FORMAT_R32F;
		case 2: return TextureFormat::FORMAT_RG32F;
		case 3: return TextureFormat::FORMAT_RGB32F;
		case 4: return TextureFormat::FORMAT_RGBA32F;
		default: return TextureFormat::FORMAT_NUM;
	}
}

// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
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
	(void)ext;
	// Per default, stb_image pretends like it can read everything
	return true;
}

Boolean load_texture(const char* path, TextureData* texData) {
	CHECK_NULLPTR(path, "texture path", false);
	CHECK_NULLPTR(path, "texture return data", false);

	// Code taken from ImageViewer
	stbi_set_flip_vertically_on_load(true);
	int width = 0;
	int height = 0;
	int components = 0;
	std::size_t bytes = 0u;
	void* data = nullptr;
	// Load either LDR or HDR
	if(stbi_is_hdr(path)) {
		data = reinterpret_cast<char*>(stbi_loadf(path, &width, &height, &components, 0));
		bytes = sizeof(float);
	} else {
		data = reinterpret_cast<char*>(stbi_load(path, &width, &height, &components, 0));
		bytes = sizeof(char);
	}

	if(data == nullptr || width <= 0 || height <= 0 || components <= 0) {
		// Fail silently since STB doesn't let us query supported extensions
		return false;
	}

	// Copy over the image data
	texData->format = get_float_format(components);
	bytes *= width * height * components;
	texData->data = new uint8_t[bytes];
	std::memcpy(texData->data, data, bytes);
	stbi_image_free(data);

	texData->width = static_cast<uint32_t>(width);
	texData->height = static_cast<uint32_t>(height);
	texData->components = static_cast<uint32_t>(components);
	texData->layers = 1u;
	return true;
}
