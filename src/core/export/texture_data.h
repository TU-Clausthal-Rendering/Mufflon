#pragma once

extern "C" {

#include <stdint.h>

typedef enum {
	FORMAT_R8U,
	FORMAT_RG8U,
	FORMAT_RGB8U,
	FORMAT_RGBA8U,
	FORMAT_R16U,
	FORMAT_RG16U,
	FORMAT_RGB16U,
	FORMAT_RGBA16U,
	FORMAT_R32F,
	FORMAT_RG32F,
	FORMAT_RGB32F,
	FORMAT_RGBA32F,
	FORMAT_RGB9E5,
	FORMAT_NUM
} TextureFormat;

typedef struct {
	uint8_t* data = nullptr;
	uint32_t width = 0u;
	uint32_t height = 0u;
	uint32_t components = 0u;
	uint32_t layers = 0u;
	TextureFormat format = TextureFormat::FORMAT_NUM;
} TextureData;

}