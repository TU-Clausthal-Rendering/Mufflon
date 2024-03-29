#pragma once

#include "core/export/texture_data.h"

#ifdef _MSC_VER
#    define EXPORT __declspec(dllexport)
#    define CDECL __cdecl
#elif defined(_WIN32)
#    define EXPORT
#    define CDECL __attribute__((__cdecl__))
#else
#    define EXPORT
#    define CDECL
#endif // _MSC_VER

extern "C" {

#include <stdint.h>

typedef uint32_t Boolean;

EXPORT Boolean CDECL can_load_texture_format(const char* ext);
EXPORT Boolean CDECL can_store_texture_format(const char* ext);
EXPORT Boolean CDECL load_texture(const char* path, TextureData* texData);
EXPORT Boolean CDECL store_texture(const char* path, const TextureData* texData);

}