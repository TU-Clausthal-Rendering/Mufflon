#pragma once

#include "api.h"
#include "core/export/texture_data.h"
#include <glad/glad.h>
// Undefine windows header macros
#undef ERROR

extern "C" {

#include <stdint.h>

// Typedef for boolean value (since the standard doesn't specify
// its size
typedef uint32_t Boolean;

typedef enum {
	LOG_PEDANTIC,
	LOG_INFO,
	LOG_WARNING,
	LOG_ERROR,
	LOG_FATAL_ERROR
} LogLevel;

OPENGLDISPLAY_API void CDECL opengldisplay_set_gamma(float val);
OPENGLDISPLAY_API float CDECL opengldisplay_get_gamma();
OPENGLDISPLAY_API void CDECL opengldisplay_set_factor(float val);
OPENGLDISPLAY_API float CDECL opengldisplay_get_factor();
OPENGLDISPLAY_API const char* CDECL opengldisplay_get_dll_error();
OPENGLDISPLAY_API Boolean CDECL opengldisplay_display(int left, int right, int bottom, int top, uint32_t width, uint32_t height);
OPENGLDISPLAY_API Boolean CDECL opengldisplay_resize_screen(uint32_t width, uint32_t height, TextureFormat format);
OPENGLDISPLAY_API Boolean CDECL opengldisplay_write(const char* data);
OPENGLDISPLAY_API Boolean CDECL opengldisplay_initialize();
OPENGLDISPLAY_API Boolean CDECL opengldisplay_set_logger(void(*logCallback)(const char*, int));
OPENGLDISPLAY_API void CDECL opengldisplay_destroy();
OPENGLDISPLAY_API Boolean CDECL opengldisplay_set_log_level(LogLevel level);

} // extern "C"