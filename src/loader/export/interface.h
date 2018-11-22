#pragma once
#include "export/api.hpp"

extern "C" {
#include <stdint.h>

typedef uint32_t Boolean;

LOADER_API Boolean CDECL loader_load_json(const char* path);

} // extern "C"

// TODO: C interface