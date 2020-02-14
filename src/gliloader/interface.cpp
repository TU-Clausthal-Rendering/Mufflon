#include "plugin/texture_plugin_interface.h"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include "util/pixel_conversion.hpp"
#include "util/string_view.hpp"
#include <gli/gl.hpp>
#include <gli/gli.hpp>
#include <ei/conversions.hpp>
#include <cuda_fp16.h>
#include <cstring>
#include <mutex>
#include <tuple>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

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

Boolean load_exr(const char* path, TextureData* texData) {
	int width, height;
	float* data;
	const char* err;
	if(int ret; ret = LoadEXR(&data, &width, &height, path, &err) != TINYEXR_SUCCESS) {
		logError("[", FUNCTION_NAME, "] Failed to load EXR file '", path, "': ", err);
		return false;
	}

	texData->components = 4u;
	texData->format = TextureFormat::FORMAT_RGBA32F;
	texData->layers = 1u;
	texData->sRgb = false;
	texData->width = static_cast<std::uint32_t>(width);
	texData->height = static_cast<std::uint32_t>(height);
	texData->data = reinterpret_cast<std::uint8_t*>(data);

	// Flip image along x-axis
	for(std::uint32_t y = 0u; y < texData->height / 2u; ++y) {
		for(std::uint32_t x = 0u; x < texData->width; ++x) {
			float tmp[4u];
			const auto index = sizeof(tmp) * (x + y * texData->width);
			const auto flippedIndex = sizeof(tmp) * (x + (texData->height - y - 1u) * texData->width);
			std::memcpy(tmp, texData->data + index, sizeof(tmp));
			std::memcpy(texData->data + index, texData->data + flippedIndex, sizeof(tmp));
			std::memcpy(texData->data + flippedIndex, tmp, sizeof(tmp));
		}
	}

	return true;
}

Boolean store_exr(const char* path, const TextureData* texData) {
	const float* data = nullptr;
	const auto channelType = util::get_channel_type(texData->format);

	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);

	// To make things easier we only export float EXRs, since we have to convert to BGR anyway
	const auto originalChannelCount = util::get_channel_count(texData->format);
	const auto pixelCount = static_cast<std::size_t>(texData->width) * static_cast<std::size_t>(texData->height);
	std::vector<float> pixels(originalChannelCount * pixelCount);
	const auto pixelSize = util::get_format_size(texData->format);
	for(std::size_t y = 0u; y < static_cast<std::size_t>(texData->height); ++y) {
		for(std::size_t x = 0u; x < static_cast<std::size_t>(texData->width); ++x) {
			const auto index = x + y * static_cast<std::size_t>(texData->width);
			const auto byteOffset = pixelSize * index;
			const auto value = util::read_pixel(reinterpret_cast<const char*>(texData->data) + byteOffset,
												TextureFormat::FORMAT_RGBA32F);
			// EXR is Y-inverted compared to our coordinate system
			const auto pixelIndex = x + (static_cast<std::size_t>(texData->height) - y - 1u)
				* static_cast<std::size_t>(texData->width);
			switch(originalChannelCount) {
				default:
				case 4:
					pixels[3u * pixelCount + pixelIndex] = value.a;
					[[fallthrough]];
				case 3:
					pixels[2u * pixelCount + pixelIndex] = value.b;
					[[fallthrough]];
				case 2:
					pixels[1u * pixelCount + pixelIndex] = value.g;
					[[fallthrough]];
				case 1:
					pixels[0u * pixelCount + pixelIndex] = value.r;
					break;
			}
		}
	}

	// It is perfectly valid to create these (possibly invalid) pointers, since
	// we never access them if the channel count does not match
	const float* pixelPtrs[] = {
		pixels.data() + 0u * pixelCount,
		pixels.data() + 1u * pixelCount,
		pixels.data() + 2u * pixelCount,
		pixels.data() + 3u * pixelCount
	};
	image.num_channels = texData->components;
	image.images = (unsigned char**)pixelPtrs;
	image.width = static_cast<int>(texData->width);
	image.height = static_cast<int>(texData->height);

	header.num_channels = std::min(4, static_cast<int>(originalChannelCount));
	auto channels = std::make_unique<EXRChannelInfo[]>(header.num_channels);
	header.channels = channels.get();
	// Must be BGR(A) order, since most of EXR viewers expect this channel order.
	switch(originalChannelCount) {
		default:
		case 4:
			strncpy(header.channels[3].name, "A", 255u); header.channels[3].name[strlen("A")] = '\0';
			[[fallthrough]];
		case 3:
			strncpy(header.channels[2].name, "B", 255u); header.channels[2].name[strlen("B")] = '\0';
			[[fallthrough]];
		case 2:
			strncpy(header.channels[1].name, "G", 255u); header.channels[1].name[strlen("G")] = '\0';
			[[fallthrough]];
		case 1:
			strncpy(header.channels[0].name, "R", 255u); header.channels[0].name[strlen("R")] = '\0';
			break;
	}

	auto pixelTypes = std::make_unique<int[]>(2 * header.num_channels);

	header.pixel_types = pixelTypes.get();
	header.requested_pixel_types = &pixelTypes[header.num_channels];
	for(int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
	}

	const char* err;
	int ret = SaveEXRImageToFile(&image, &header, path, &err);
	if(ret != TINYEXR_SUCCESS) {
		logError("[", FUNCTION_NAME, "] Failed to save EXR file '", path, "': ", err);
		return false;
	}

	return true;
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
		case gli::format::FORMAT_R16_SFLOAT_PACK16: texData.format = TextureFormat::FORMAT_RG16F; texData.sRgb = false; texData.components = 1u; return 1u;
		case gli::format::FORMAT_RG16_SFLOAT_PACK16: texData.format = TextureFormat::FORMAT_RG16F; texData.sRgb = false; texData.components = 2u; return 2u;
		case gli::format::FORMAT_RGB16_SFLOAT_PACK16: texData.format = TextureFormat::FORMAT_RG16F; texData.sRgb = false; texData.components = 4u; return 3u;
		case gli::format::FORMAT_RGBA16_SFLOAT_PACK16: texData.format = TextureFormat::FORMAT_RGBA16F; texData.sRgb = false; texData.components = 4u; return 4u;
		case gli::format::FORMAT_R32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_R32F; texData.sRgb = false; texData.components = 1u;  return 1u;
		case gli::format::FORMAT_RG32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RG32F; texData.sRgb = false; texData.components = 2u;  return 2u;
		case gli::format::FORMAT_RGB32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA32F; texData.sRgb = false; texData.components = 4u;  return 3u;
		case gli::format::FORMAT_RGBA32_SFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA32F; texData.sRgb = false; texData.components = 4u;  return 4u;
		case gli::format::FORMAT_RGB9E5_UFLOAT_PACK32: texData.format = TextureFormat::FORMAT_RGBA16F; texData.sRgb = false; texData.components = 4u;  return 3u;
		default: return 0u;
	}
}

gli::format get_format(const TextureData& texData) {
	switch(texData.format) {
	case TextureFormat::FORMAT_R8U: return texData.sRgb ? gli::format::FORMAT_R8_SRGB_PACK8 : gli::format::FORMAT_R8_UINT_PACK8;
	case TextureFormat::FORMAT_RG8U: return texData.sRgb ? gli::format::FORMAT_RG8_SRGB_PACK8 : gli::format::FORMAT_RG8_UINT_PACK8;
	case TextureFormat::FORMAT_RGBA8U: return texData.sRgb ? gli::format::FORMAT_RGBA8_SRGB_PACK8 : gli::format::FORMAT_RGBA8_UINT_PACK8;
	case TextureFormat::FORMAT_R16U: return gli::format::FORMAT_R16_UINT_PACK16;
	case TextureFormat::FORMAT_RG16U: return gli::format::FORMAT_RG16_UINT_PACK16;
	case TextureFormat::FORMAT_RGBA16U: return gli::format::FORMAT_RGBA16_UINT_PACK16;
	case TextureFormat::FORMAT_R32F: return gli::format::FORMAT_R32_SFLOAT_PACK32;
	case TextureFormat::FORMAT_RG32F: return gli::format::FORMAT_RG32_SFLOAT_PACK32;
	case TextureFormat::FORMAT_RGBA32F: return gli::format::FORMAT_RGBA32_SFLOAT_PACK32;
	case TextureFormat::FORMAT_R16F: return gli::format::FORMAT_R16_SFLOAT_PACK16;
	case TextureFormat::FORMAT_RG16F: return gli::format::FORMAT_RG16_SFLOAT_PACK16;
	case TextureFormat::FORMAT_RGBA16F: return gli::format::FORMAT_RGBA16_SFLOAT_PACK16;
	default:
		throw std::runtime_error("Unsupported texture format");
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

Boolean can_load_texture_format(const char* ext) {
	return std::strncmp(ext, ".dds", 4u) == 0
		|| std::strncmp(ext, ".ktx", 4u) == 0
		|| std::strncmp(ext, ".exr", 4u) == 0;
}

Boolean can_store_texture_format(const char* ext) {
	return std::strncmp(ext, ".dds", 4u) == 0
		|| std::strncmp(ext, ".ktx", 4u) == 0
		|| std::strncmp(ext, ".exr", 4u) == 0;
}


Boolean load_texture(const char* path, TextureData* texData) {
	try {
		CHECK_NULLPTR(path, "texture path", false);
		CHECK_NULLPTR(texData, "texture return data", false);

		StringView pathView = path;
		if(!fs::exists(path)) {
			logError("[", FUNCTION_NAME, "] Texture file '", pathView, "' does not exist");
			return false;
		}

		// Check if we need to load OpenEXR
		if(pathView.length() >= 4u && pathView.substr(pathView.length() - 4u).compare(".exr") == 0)
			return load_exr(path, texData);

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
					const ei::uint32* data = reinterpret_cast<const ei::uint32*>(tex.data(layer, face, mipmap));
					__half* layerData = reinterpret_cast<__half*>(&texData->data[(face + layer * tex.faces()) * layerSize]);
					for(std::size_t y = 0u; y < texData->height; ++y) {
						for(std::size_t x = 0u; x < texData->width; ++x) {
							ei::Vec3 rgb = ei::unpackRGB9E5(data[y * texData->width + x]);
							// We need to flip the texture on load
							const std::size_t pixel = channels * ((texData->height - y - 1u) * texData->width + x);
							layerData[pixel + 0u] = __float2half(rgb.r);
							layerData[pixel + 1u] = __float2half(rgb.g);
							layerData[pixel + 2u] = __float2half(rgb.b);
							layerData[pixel + 3u] = __float2half(0.f);
						}
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
					for(std::size_t y = 0u; y < texData->height; ++y) {
						for(std::size_t x = 0u; x < texData->width; ++x) {
							// Swap image vertically
							const std::size_t origPixel = origTexelSize * (y * texData->width + x);
							const std::size_t pixel = texelSize * ((texData->height - y - 1u)* texData->width + x);
							if(origChannels == channels) {
								std::memcpy(&layerData[pixel], &data[origPixel], origTexelSize);
							} else {
								std::memcpy(&layerData[pixel], &data[origPixel], origTexelSize);
								std::memset(&layerData[pixel + origTexelSize], 0, componentSize);
							}
						}
					}
				}
			}
		}

		return true;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture load for '",
				 path, "' caught exception: ", e.what());
		if(texData->data)
			delete[] texData->data;
		return false;
	}
}

Boolean store_texture(const char* path, const TextureData* texData) {
	try {
		CHECK_NULLPTR(path, "texture path", false);
		CHECK_NULLPTR(texData, "texture return data", false);

		StringView pathView = path;

		// Check if we need to store OpenEXR
		if(pathView.length() >= 4u && pathView.substr(pathView.length() - 4u).compare(".exr") == 0)
			return store_exr(path, texData);

		constexpr std::size_t mipmap = 0u;

		gli::texture::extent_type extent(texData->width, texData->height, 1);
		gli::texture tex;
		if(texData->layers == 6) {
			tex = gli::texture(gli::TARGET_CUBE, get_format(*texData), extent, 1, texData->layers, 1);
		} else {
			tex = gli::texture(gli::TARGET_2D, get_format(*texData), extent, 1, texData->layers, 1);
		}
		const std::size_t channels = get_channels(texData->format);
		const std::size_t componentSize = get_component_size(texData->format);
		const std::size_t texelSize = channels * componentSize;
		const std::size_t layerSize = texData->width * texData->height * texelSize;
		for(std::size_t layer = 0u; layer < tex.layers(); ++layer) {
			for(std::size_t face = 0u; face < tex.faces(); ++face) {
				const uint8_t* layerData = &texData->data[(face + layer * tex.faces()) * layerSize];
				char* data = reinterpret_cast<char*>(tex.data(layer, face, mipmap));
				for(std::size_t y = 0u; y < texData->height; ++y) {
					for(std::size_t x = 0u; x < texData->width; ++x) {
						// Swap image vertically
						const std::size_t origPixel = texelSize * (y * texData->width + x);
						const std::size_t pixel = texelSize * ((texData->height - y - 1u)* texData->width + x);
						std::memcpy(&data[origPixel], &layerData[pixel], texelSize);
					}
				}
			}
		}
		gli::save(tex, pathView.data());
		return true;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture store for '",
				 path, "' caught exception: ", e.what());
		return false;
	}
}