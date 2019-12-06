#pragma once

#include "mff_api.h"
#include "core_interface.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#include <stdint.h>

typedef uint32_t Boolean;

typedef enum {
	LOADER_SUCCESS,
	LOADER_ERROR,
	LOADER_ABORT
} LoaderStatus;

LOADER_API const char* CDECL loader_get_dll_error();;
LOADER_API LoaderStatus CDECL loader_load_json(const char* path);
LOADER_API LoaderStatus CDECL loader_save_scene(const char* path);
LOADER_API Boolean CDECL loader_load_lod(void* obj, uint32_t lod);
LOADER_API Boolean CDECL loader_abort();
LOADER_API const char* CDECL loader_get_loading_status();
LOADER_API void CDECL loader_profiling_enable();
LOADER_API void CDECL loader_profiling_disable();
LOADER_API Boolean CDECL loader_profiling_set_level(ProfilingLevel level);
LOADER_API Boolean CDECL loader_profiling_save_current_state(const char* path);
LOADER_API Boolean CDECL loader_profiling_save_snapshots(const char* path);
LOADER_API Boolean CDECL loader_profiling_save_total_and_snapshots(const char* path);
LOADER_API const char* CDECL loader_profiling_get_current_state();
LOADER_API const char* CDECL loader_profiling_get_snapshots();
LOADER_API const char* CDECL loader_profiling_get_total();
LOADER_API const char* CDECL loader_profiling_get_total_and_snapshots();
LOADER_API void CDECL loader_profiling_reset();

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// TODO: C interface
