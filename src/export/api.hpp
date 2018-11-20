#pragma once

#ifdef _WIN32
#    ifdef CORE_EXPORTS
#        define CORE_API __declspec(dllexport)
#    else
#        define CORE_API __declspec(dllimport)
#    endif
#    ifdef LOADER_EXPORTS
#        define LOADER_API __declspec(dllexport)
#    else
#        define LOADER_API __declspec(dllimport)
#    endif
#    define CDECL __cdecl
#elif
#    define CORE_API
#    define LOADER_API
#    define CDECL __attribute__((__cdecl__))
#endif

#ifdef __CUDACC__
#define CUDA_FUNCTION inline __host__ __device__
#else
#define CUDA_FUNCTION inline
#endif // __CUDACC__