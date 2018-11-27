#pragma once

#include "export/api.hpp"
#include "core/export/texture_data.h"

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else // _MSC_VER
#define EXPORT
#endif // _MSC_VER

extern "C" {

#include <stdint.h>

typedef uint32_t Boolean;

EXPORT Boolean CDECL can_load_texture_format(const char* ext);
EXPORT Boolean CDECL load_texture(const char* path, TextureData* texData);

}