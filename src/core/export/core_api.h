#pragma once

#ifdef _WIN32
#    ifdef CORE_EXPORTS
#        define CORE_API __declspec(dllexport)
#    else
#        define CORE_API __declspec(dllimport)
#    endif
#else
#    define CORE_API
#endif

#ifdef _MSC_VER
#    define CDECL __cdecl
#elif defined(_WIN32)
#    define CDECL __attribute__((__cdecl__))
#else
#    define CDECL
#endif // _MSC_VER

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#else // __CUDACC__
#define CUDA_FUNCTION
#endif