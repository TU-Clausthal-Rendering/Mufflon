#pragma once

#ifdef _WIN32
#    ifdef OPENGLDISPLAY_EXPORTS
#        define OPENGLDISPLAY_API __declspec(dllexport)
#    else
#        define OPENGLDISPLAY_API __declspec(dllimport)
#    endif
#elif
#    define OPENGLDISPLAY_API
#endif

#ifdef _MSC_VER
#    define CDECL __cdecl
#else
#    define CDECL __attribute__((__cdecl__))
#endif // _MSC_VER
