#include "plugin/texture_interface.h"
#include "util/log.hpp"
#include <fstream>
#include <stdexcept>
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

template < class T >
T read(std::istream& stream) {
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
	return std::strncmp(ext, ".pfm", 4u) == 0u;
}

Boolean load_texture(const char* path, TextureData* texData) {
	CHECK_NULLPTR(path, "texture path", false);
	CHECK_NULLPTR(path, "texture return data", false);

	// Code taken from ImageViewer
	std::ifstream stream(path, std::ios::binary);
	if(!stream.is_open()) {
		logError("[", FUNCTION_NAME, "] Could not open texture file '",
				 path, "'");
		return false;
	}
	try {
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
			texData->format = TextureFormat::FORMAT_RGB32F;
		texData->width = static_cast<std::uint32_t>(width);
		texData->height = static_cast<std::uint32_t>(height);
		texData->components = grayscale ? 1u : 3u;
		texData->layers = 1u;
		texData->sRgb = 0u;
		float* data = new float[texData->width * texData->height * texData->components];

		if(std::strncmp(bands, "Pf", 2u) == 0) {
			for(std::uint32_t y = 0u; y < texData->height; ++y) {
				for(std::uint32_t x = 0u; x < texData->width; ++x) {
					float val = read<float>(stream);
					if(needsByteSwap)
						val = swap_bytes(val);
					data[y * texData->width + x] = val;
				}
			}
		} else if(std::strncmp(bands, "PF", 2u) == 0) {
			for(std::uint32_t y = 0u; y < texData->height; ++y) {
				for(std::uint32_t x = 0u; x < texData->width; ++x) {
					float r = read<float>(stream);
					float g = read<float>(stream);
					float b = read<float>(stream);
					if(needsByteSwap) {
						r = swap_bytes(r);
						g = swap_bytes(g);
						b = swap_bytes(b);
					}
					data[3u * (y * texData->width + x) + 0u] = r;
					data[3u * (y * texData->width + x) + 1u] = g;
					data[3u * (y * texData->width + x) + 2u] = b;
				}
			}
		}

		texData->data = reinterpret_cast<uint8_t*>(data);
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Texture load for '",
				 path, "' caught exception: ", e.what());
		return false;
	}
	return true;
}
