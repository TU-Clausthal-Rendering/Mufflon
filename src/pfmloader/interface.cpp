#include "plugin/texture_plugin_interface.h"
#include "util/log.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <ei/vector.hpp>

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

template < class T >
T read(std::istream& stream) {
	T val;
	stream >> val;
	return val;
}

template < class T >
T read_bytes(std::istream& stream) {
	T val;
	stream.read(reinterpret_cast<char*>(&val), sizeof(T));
	return val;
}

void skip_spaces(std::istream& stream) {
	char c;
	do
		c = stream.get();
	while(c == '\n' || c == ' ' || c == '\t' || c == '\r');
	stream.unget();
}

bool is_little_endian_machine() {
	std::uint32_t intval = 1u;
	return reinterpret_cast<unsigned char*>(&intval)[0u] == 1u;
}

template < class T >
constexpr T swap_bytes(T val) {
	T res;
	for(std::size_t i = 0u; i < sizeof(T); ++i)
		reinterpret_cast<char*>(&res)[i] = reinterpret_cast<const char*>(&val)[sizeof(T) - i - 1u];
	return res;
}

} // namespace

Boolean can_load_texture_format(const char* ext) {
	return std::strncmp(ext, ".pfm", 4u) == 0u;
}

Boolean can_store_texture_format(const char* ext) {
	return std::strncmp(ext, ".pfm", 4u) == 0u;
}

Boolean load_texture(const char* path, TextureData* texData) {
	try {
		CHECK_NULLPTR(path, "texture path", false);
		CHECK_NULLPTR(texData, "texture return data", false);

		// Code taken from ImageViewer
		std::ifstream stream(path, std::ios::binary);
		if(!stream.is_open()) {
			logError("[", FUNCTION_NAME, "] Could not open texture file '",
					 path, "'");
			return false;
		}
		stream.exceptions(std::ios::failbit);

		char bands[2u];
		stream.read(bands, 2u);
		skip_spaces(stream);

		if(std::strncmp(bands, "Pf", 2u) != 0
		   && std::strncmp(bands, "PF", 2u) != 0)
			throw std::runtime_error("Unknown bands description '"
									 + std::string(bands, 2u) + "'");

		const int width = read<int>(stream);
		const int height = read<int>(stream);
		const float scalef = read<float>(stream);
		const bool needsByteSwap = is_little_endian_machine() != (scalef < 0.f);

		if(width <= 0 || height <= 0)
			throw std::runtime_error("Invalid width/height (< 0)");

		// Read a single newline
		char c = stream.get();
		if(c == '\r')
			c = stream.get();	// ... except if there's a carriage return
		if(c != '\n')
			throw std::runtime_error("Expected newline in header");

		const bool grayscale = std::strncmp(bands, "Pf", 2u) == 0;
		if(grayscale)
			texData->format = TextureFormat::FORMAT_R32F;
		else
			texData->format = TextureFormat::FORMAT_RGBA32F;
		texData->width = static_cast<std::uint32_t>(width);
		texData->height = static_cast<std::uint32_t>(height);
		texData->components = grayscale ? 1u : 4u;
		texData->layers = 1u;
		texData->sRgb = 0u;
		float* data = new float[texData->width * texData->height * texData->components];

		if(std::strncmp(bands, "Pf", 2u) == 0) {
			for(std::uint32_t y = 0u; y < texData->height; ++y) {
				for(std::uint32_t x = 0u; x < texData->width; ++x) {
					float val = read_bytes<float>(stream);
					if(needsByteSwap)
						val = swap_bytes(val);
					data[y * texData->width + x] = val;
				}
			}
		} else if(std::strncmp(bands, "PF", 2u) == 0) {
			for(std::uint32_t y = 0u; y < texData->height; ++y) {
				for(std::uint32_t x = 0u; x < texData->width; ++x) {
					float r = read_bytes<float>(stream);
					float g = read_bytes<float>(stream);
					float b = read_bytes<float>(stream);
					if(needsByteSwap) {
						r = swap_bytes(r);
						g = swap_bytes(g);
						b = swap_bytes(b);
					}
					data[4u * (y * texData->width + x) + 0u] = r;
					data[4u * (y * texData->width + x) + 1u] = g;
					data[4u * (y * texData->width + x) + 2u] = b;
					data[4u * (y * texData->width + x) + 3u] = 0u;
				}
			}
		}

		texData->data = reinterpret_cast<uint8_t*>(data);
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

		const int numChannels = texData->components;

		std::ofstream file(path, std::ofstream::binary | std::ofstream::out);
		if(file.bad()) {
			throw std::runtime_error(" Failed to open screenshot file '" + std::string(path) + "'");
		}
		if(numChannels == 1)
			file.write("Pf\n", 3);
		else
			file.write("PF\n", 3);

		auto sizes = std::to_string(texData->width) + " " + std::to_string(texData->height);
		file.write(sizes.c_str(), sizes.length());
		file.write("\n-1.000000\n", 11);

		const auto pixels = reinterpret_cast<const char *>(texData->data);
		if(texData->format == TextureFormat::FORMAT_R32F || texData->format == TextureFormat::FORMAT_RG32F
			|| texData->format == TextureFormat::FORMAT_RGB32F || texData->format == TextureFormat::FORMAT_RGBA32F) {
			for(uint32_t y = 0; y < texData->height; ++y) {
				for(uint32_t x = 0; x < texData->width; ++x) {
					// PFM can only store one or three channel textures.
					// Write RG, RGB and RGBA as RGB and R as R.
					switch(numChannels) {
						case 1: file.write(&pixels[(y * texData->width + x) * sizeof(float)], sizeof(float)); break;
						default: {
							ei::Vec4 pixel { 0.0f };
							memcpy(&pixel, &pixels[(y * texData->width + x) * numChannels * sizeof(float)], numChannels * sizeof(float));
							file.write(reinterpret_cast<const char*>(&pixel), 3u * sizeof(float));
						}
					}
				}
			}
		} else {
			throw std::runtime_error("Non-float formats are not supported yet");
		}
		return true;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture store for '",
			path, "' caught exception: ", e.what());
		return false;
	}

}