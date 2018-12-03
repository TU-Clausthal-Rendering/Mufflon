#include "plugin/texture_interface.h"
#include "util/log.hpp"
#include <gli/gl.hpp>
#include <gli/gli.hpp>
#include <ei/conversions.hpp>
#include <cuda_fp16.h>
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

unsigned char get_format(gli::format format, TextureData& texData) {
	switch(format) {
		case gli::format::FORMAT_R8_UINT_PACK8: texData.format = TextureFormat::FORMAT_R8U; texData.sRgb = false; texData.components = 1u;  return 1u;
		case gli::format::FORMAT_R8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_R8U; texData.sRgb = true; texData.components = 1u;  return 1u;
		case gli::format::FORMAT_RG8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RG8U; texData.sRgb = false; texData.components = 2u;  return 2u;
		case gli::format::FORMAT_RG8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RG8U; texData.sRgb = true; texData.components = 2u;  return 2u;
		case gli::format::FORMAT_RGB8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = false; texData.components = 4u;  return 3u;
		case gli::format::FORMAT_RGB8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = true; texData.components = 4u;  return 3u;
		case gli::format::FORMAT_RGBA8_UINT_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = false; texData.components = 4u;  return 4u;
		case gli::format::FORMAT_RGBA8_SRGB_PACK8: texData.format = TextureFormat::FORMAT_RGBA8U; texData.sRgb = true; texData.components = 4u;  return 4u;
		case gli::format::FORMAT_R16_UINT_PACK16: texData.format = TextureFormat::FORMAT_R16U; texData.sRgb = false; texData.components = 1u;  return 1u;
		case gli::format::FORMAT_RG16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RG16U; texData.sRgb = false; texData.components = 2u;  return 2u;
		case gli::format::FORMAT_RGB16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RGBA16U; texData.sRgb = false; texData.components = 4u;  return 3u;
		case gli::format::FORMAT_RGBA16_UINT_PACK16: texData.format = TextureFormat::FORMAT_RGBA16U; texData.sRgb = false; texData.components = 4u;  return 4u;
		case gli::format::FORMAT_R32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_R32F; texData.sRgb = false; texData.components = 1u;  return 1u;
		case gli::format::FORMAT_RG32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RG32F; texData.sRgb = false; texData.components = 2u;  return 2u;
		case gli::format::FORMAT_RGB32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA32F; texData.sRgb = false; texData.components = 4u;  return 3u;
		case gli::format::FORMAT_RGBA32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA32F; texData.sRgb = false; texData.components = 4u;  return 4u;
		case gli::format::FORMAT_RGB9E5_UFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA16F; texData.sRgb = false; texData.components = 4u;  return 3u;
		default: return 0u;
	}
}

std::size_t get_component_size(TextureFormat format) {
	switch(format) {
		case TextureFormat::FORMAT_R8U:
		case TextureFormat::FORMAT_RG8U:
		case TextureFormat::FORMAT_RGBA8U:
			return 1u;
		case TextureFormat::FORMAT_R16U:
		case TextureFormat::FORMAT_RG16U:
		case TextureFormat::FORMAT_RGBA16U:
		case TextureFormat::FORMAT_R16F:
		case TextureFormat::FORMAT_RG16F:
		case TextureFormat::FORMAT_RGBA16F:
			return 2u;
		case TextureFormat::FORMAT_R32F:
		case TextureFormat::FORMAT_RG32F:
		case TextureFormat::FORMAT_RGBA32F:
			return 4u;
		default: return 0u;
	}
}

std::size_t get_channels(TextureFormat format) {
	switch(format) {
		case TextureFormat::FORMAT_R8U:
		case TextureFormat::FORMAT_R16U:
		case TextureFormat::FORMAT_R16F:
		case TextureFormat::FORMAT_R32F:
			return 1u;
		case TextureFormat::FORMAT_RG8U:
		case TextureFormat::FORMAT_RG16U:
		case TextureFormat::FORMAT_RG16F:
		case TextureFormat::FORMAT_RG32F:
			return 2u;
		case TextureFormat::FORMAT_RGBA8U:
		case TextureFormat::FORMAT_RGBA16U:
		case TextureFormat::FORMAT_RGBA16F:
		case TextureFormat::FORMAT_RGBA32F:
			return 4u;
		default: return 0u;
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

	// For us, a cubemap is just a texture array with 6 faces
	if(tex.faces() != 1u) {
		if(tex.layers() != 1u) {
			logError("[", FUNCTION_NAME, "] Cannot have cubemap texture arrays: '", pathView, "'");
			return false;
		}
		texData->layers = static_cast<std::uint32_t>(tex.faces());
	} else {
		texData->layers = static_cast<std::uint32_t>(tex.layers());
	}
	constexpr std::size_t mipmap = 0u;
	texData->width = static_cast<std::uint32_t>(tex.extent(mipmap).x);
	texData->height = static_cast<std::uint32_t>(tex.extent(mipmap).y);
	// Determine the format
	std::size_t origChannels = get_format(tex.format(), *texData);
	if(origChannels == 0u) {
		logError("[", FUNCTION_NAME, "] Unknown texture format of texture '", pathView, "'");
		return false;
	}
	if(tex.extent(mipmap).z != 1u) {
		logError("[", FUNCTION_NAME, "] 3D textures are not supported: '", pathView, "'");
		return false;
	}
	if(gli::is_compressed(tex.format())) {
		logError("[", FUNCTION_NAME, "] Compressed textures are not supported: '", pathView, "'");
		return false;
	}

	if(tex.format() == gli::format::FORMAT_RGB9E5_UFLOAT_PACK32) {
		// We decode this format into (half-)floats
		static_assert(sizeof(__half) == 2u, "Invalid assumption about half-float size");

		constexpr std::size_t channels = 4u;
		constexpr std::size_t channelSize = 2u;
		constexpr std::size_t texelSize = channels * channelSize;
		const std::size_t layerSize = texData->width * texData->height * texelSize;
		texData->data = new std::uint8_t[texData->layers * layerSize];
		for(std::size_t layer = 0u; layer < tex.layers(); ++layer) {
			for(std::size_t face = 0u; face < tex.faces(); ++face) {
				const unsigned char* data = reinterpret_cast<const unsigned char*>(tex.data(layer, face, mipmap));
				//float* layerData = reinterpret_cast<float*>(&texData->data[(face + layer * tex.faces()) * layerSize]);
				__half* layerData = reinterpret_cast<__half*>(&texData->data[(face + layer * tex.faces()) * layerSize]);
				for(std::size_t b = 0u; b < texData->width * texData->height; ++b) {
					eitypes::uint32 rgb9e5;
					std::memcpy(&rgb9e5, &data[4u * b], 4u);
					ei::Vec3 rgb = ei::unpackRGB9E5(rgb9e5);
					layerData[channels * b + 0u] = __float2half(rgb.r);
					layerData[channels * b + 1u] = __float2half(rgb.g);
					layerData[channels * b + 2u] = __float2half(rgb.b);
					layerData[channels * b + 3u] = __float2half(0.f);
				}
			}
		}
	} else {
		const std::size_t channels = get_channels(texData->format);
		const std::size_t componentSize = get_component_size(texData->format);
		const std::size_t texelSize = channels * componentSize;
		const std::size_t origTexelSize = origChannels * componentSize;
		const std::size_t layerSize = texData->width * texData->height * texelSize;
		texData->data = new std::uint8_t[texData->layers * layerSize];
		for(std::size_t layer = 0u; layer < tex.layers(); ++layer) {
			for(std::size_t face = 0u; face < tex.faces(); ++face) {
				uint8_t* layerData = &texData->data[(face + layer * tex.faces()) * layerSize];
				const char* data = reinterpret_cast<const char*>(tex.data(layer, face, mipmap));
				if(origChannels == channels) {
					// Can memcpy
					std::memcpy(layerData, data, layerSize);
				} else {
					// Need to do it texel by texel (should only happen for RGB without alpha)
					for(std::size_t b = 0u; b < texData->width * texData->height; ++b) {
						std::memcpy(&layerData[texelSize * b], &data[origTexelSize * b], origTexelSize);
						std::memset(&layerData[texelSize * b + origTexelSize], 0, componentSize);
					}
				}
			}
		}
	}

	return true;
}
