#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdint.h>
#include <stddef.h>

typedef enum {
	FORMAT_R8U,
	FORMAT_RG8U,
	FORMAT_RGBA8U,
	FORMAT_R16U,
	FORMAT_RG16U,
	FORMAT_RGBA16U,
	FORMAT_R16F,
	FORMAT_RG16F,
	FORMAT_RGBA16F,
	FORMAT_R32F,
	FORMAT_RG32F,
	FORMAT_RGBA32F,
	// RGB formats are not supported by the Texture class.
	// Since TextureFormat(textures::Format) should be a valid cast,
	// special in/output formats must be added at the end of this enum.
	FORMAT_RGB8U,
	FORMAT_RGB16U,
	FORMAT_RGB16F,
	FORMAT_RGB32F,
	FORMAT_NUM
} TextureFormat;

typedef struct {
	// A block of data which is formated with .format (on padding or similar).
	uint8_t* data;
	uint32_t width;
	uint32_t height;
	uint32_t components;
	uint32_t layers;
	TextureFormat format;
	uint32_t sRgb;
} TextureData;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus