#include "plugin/texture_interface.h"
#include "util/log.hpp"
#include <gli/gl.hpp>
#include <gli/gli.hpp>

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

Boolean load_openexr(const char* path, TextureData* texData) {
	// TODO
	return false;
}

} // namespace


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
	
	// TODO: determine format

	texData->layers = static_cast<std::uint32_t>(tex.layers());
	for(std::uint32_t layer = 0u; layer < texData->layers; ++layer) {
		// TODO: load layers
		for(std::size_t face = 0u; face < tex.faces(); ++face) {
			// TODO: load face
			for(std::size_t mip = 0u; mip < tex.levels(); ++mip) {
				// TODO: load mip-map
			}
		}
	}

	return false;
}
