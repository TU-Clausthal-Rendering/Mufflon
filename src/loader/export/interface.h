#pragma once
#include "export/api.hpp"

extern "C" {

LOADER_API bool CDECL load_scene_file(const char* path);

} // extern "C"

// TODO: C interface