#pragma once

#ifdef _WIN32
#    ifdef LOADER_EXPORTS
#        define LOADER_API __declspec(dllexport)
#    else
#        define LOADER_API __declspec(dllimport)
#    endif
#elif
#    define LOADER_API
#endif

#ifdef _MSC_VER
#    define CDECL __cdecl
#else
#    define CDECL __attribute__((__cdecl__))
#endif // _MSC_VER
