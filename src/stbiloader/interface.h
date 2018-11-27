#pragma once

#include "export/api.hpp"
#include <cstdint>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else // _MSC_VER
#define EXPORT
#endif // _MSC_VER

extern "C" {

typedef uint32_t Boolean;

EXPORT Boolean CDECL can_load_texture_format(const char* ext);
EXPORT void CDECL load_texture(const char* path);

}