#include "plugin/texture_plugin_interface.h"
#include "util/log.hpp"
#include "util/pixel_conversion.hpp"
#include <ei/conversions.hpp>
#include <cstring>
#include <filesystem>
#include <mutex>

#define STB_IMAGE_IMPLEMENTATION
#include <stbi/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stbi/stb_image_write.h>

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

TextureFormat get_int_format(int components) {
	switch(components) {
		case 1: return TextureFormat::FORMAT_R8U;
		case 2: return TextureFormat::FORMAT_RG8U;
		case 4: return TextureFormat::FORMAT_RGBA8U;
		default: return TextureFormat::FORMAT_NUM;
	}
}

TextureFormat get_float_format(int components) {
	switch(components) {
		case 1: return TextureFormat::FORMAT_R32F;
		case 2: return TextureFormat::FORMAT_RG32F;
		case 4: return TextureFormat::FORMAT_RGBA32F;
		default: return TextureFormat::FORMAT_NUM;
	}
}

} // namespace

Boolean can_load_texture_format(const char* ext) {
	return std::strncmp(ext, ".hdr", 4u) == 0
		|| std::strncmp(ext, ".bmp", 4u) == 0
		|| std::strncmp(ext, ".tga", 4u) == 0
		|| std::strncmp(ext, ".tif", 4u) == 0
		|| std::strncmp(ext, ".jpg", 4u) == 0
		|| std::strncmp(ext, ".jpeg", 5u) == 0
		|| std::strncmp(ext, ".png", 4u) == 0;
}

Boolean can_store_texture_format(const char* ext) {
	return std::strncmp(ext, ".hdr", 4u) == 0
		|| std::strncmp(ext, ".bmp", 4u) == 0
		|| std::strncmp(ext, ".tga", 4u) == 0
		|| std::strncmp(ext, ".tif", 4u) == 0
		|| std::strncmp(ext, ".jpg", 5u) == 0
		|| std::strncmp(ext, ".jpeg", 5u) == 0
		|| std::strncmp(ext, ".png", 4u) == 0;
}

Boolean load_texture(const char* path, TextureData* texData) {
	try {
		CHECK_NULLPTR(path, "texture path", false);
		CHECK_NULLPTR(path, "texture return data", false);

		// Code taken from ImageViewer
		stbi_set_flip_vertically_on_load(true);
		int width = 0;
		int height = 0;
		int components = 0;
		std::size_t perElemBytes = 0u;
		char* data = nullptr;
		// Load either LDR or HDR
		if(stbi_is_hdr(path)) {
			data = reinterpret_cast<char*>(stbi_loadf(path, &width, &height, &components, 0));
			perElemBytes = sizeof(float);
			texData->sRgb = 0u;
		} else {
			data = reinterpret_cast<char*>(stbi_load(path, &width, &height, &components, 0));
			perElemBytes = sizeof(char);
			texData->sRgb = 1u;
		}

		if(data == nullptr || width <= 0 || height <= 0 || components <= 0) {
			// Fail silently since STB doesn't let us query supported extensions
			return false;
		}

		// Make sure that we have either 1, 2, or 4 channels
		if(components == 3) {
			components = 4;
			// Copy over the image data one by one
			std::size_t bytes = perElemBytes * width * height * components;
			texData->data = new uint8_t[bytes];
			for(std::size_t t = 0u; t < static_cast<std::size_t>(width * height); ++t) {
				std::memcpy(&texData->data[components * perElemBytes * t],
							&data[3u * perElemBytes * t],
							3u * perElemBytes);
				// Ignore alpha channel
				std::memset(&texData->data[components * perElemBytes * t + 3u * perElemBytes],
							0, perElemBytes);
			}
		} else {
			// May use memcpy
			std::size_t bytes = perElemBytes * width * height * components;
			texData->data = new uint8_t[bytes];
			std::memcpy(texData->data, data, bytes);
		}

		texData->format = perElemBytes == sizeof(float) ? get_float_format(components) : get_int_format(components);
		stbi_image_free(data);

		texData->width = static_cast<uint32_t>(width);
		texData->height = static_cast<uint32_t>(height);
		texData->components = static_cast<uint32_t>(components);
		texData->layers = 1u;
		return true;
	}  catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture load for '",
				 path, "' caught exception: ", e.what());
		if(texData->data)
			delete[] texData->data;
		return false;
	} catch(...) {
		logError("[", FUNCTION_NAME, "] Texture load for '",
				 path, "' caught unknown exception");
		if(texData->data)
			delete[] texData->data;
		return false;
	}
}
Boolean store_texture(const char* path, const TextureData* texData) {
	try {
		CHECK_NULLPTR(path, "texture path", false);
		CHECK_NULLPTR(path, "texture return data", false);

		const auto filePath = std::filesystem::u8path(path);
		stbi_flip_vertically_on_write(true);
		if(filePath.extension() == ".hdr") {
			if(util::get_channel_size(texData->format) == 4u) {
				const float* data = reinterpret_cast<const float*>(texData->data);
				stbi_write_hdr(path, texData->width, texData->height, texData->components, data);
			} else {
				// Convert possibly LDR data to HDR
				auto hdrData = std::make_unique<float[]>(texData->components * texData->width * texData->height);
				const auto pixelSize = util::get_format_size(texData->format);
				for(std::uint32_t y = 0u; y < texData->height; ++y) {
					for(std::uint32_t x = 0u; x < texData->width; ++x) {
						const auto index = x + y * static_cast<std::size_t>(texData->width);
						const auto byteOffset = pixelSize * index;
						const auto value = util::read_pixel(reinterpret_cast<const char*>(texData->data) + byteOffset,
															TextureFormat::FORMAT_RGBA32F);
						for(unsigned c = 0u; c < texData->components; ++c)
							hdrData[texData->components * index + c] = value[c];
					}
				}
				stbi_write_hdr(path, texData->width, texData->height, texData->components, hdrData.get());
			}
		} else {
			if(util::get_channel_size(texData->format) == 1u) {
				stbi_write_png(path, texData->width, texData->height,
							   texData->components, texData->data, 0);
			} else {
				// Convert possibly HDR data to LDR
				auto ldrData = std::make_unique<unsigned char[]>(texData->components * texData->width * texData->height);
				const auto pixelSize = util::get_format_size(texData->format);
				for(std::uint32_t y = 0u; y < texData->height; ++y) {
					for(std::uint32_t x = 0u; x < texData->width; ++x) {
						const auto index = x + y * static_cast<std::size_t>(texData->width);
						const auto byteOffset = pixelSize * index;
						const auto value = util::read_pixel(reinterpret_cast<const char*>(texData->data) + byteOffset,
															TextureFormat::FORMAT_RGBA32F);
						for(unsigned c = 0u; c < texData->components; ++c) {
							// Bring value to sRGB
							const auto sRgb = ei::rgbToSRgb(std::min(1.f, std::max(0.f, value[c])));
							ldrData[texData->components * index + c] = static_cast<unsigned char>(sRgb * 255u);
						}
					}
				}
				stbi_write_png(path, texData->width, texData->height,
							   texData->components, ldrData.get(), 0);
			}
		}

		return true;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture store for '",
			path, "' caught exception: ", e.what());
		return false;
	}
}