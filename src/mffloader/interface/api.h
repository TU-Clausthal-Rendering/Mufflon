#pragma once

#ifdef _WIN32
#    ifdef LOADER_EXPORTS
#        define LOADER_API __declspec(dllexport)
#    else
#        define LOADER_API __declspec(dllimport)
#    endif
#else
#    define LOADER_API
#endif

#ifdef _MSC_VER
#    define CDECL __cdecl
#elif defined(_WIN32)
#    define CDECL __attribute__((__cdecl__))
#else
#    define CDECL
#endif // _MSC_VER
