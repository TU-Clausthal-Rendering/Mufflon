#pragma once

#ifdef _WIN32
#    ifdef CORE_EXPORTS
#        define CORE_API __declspec(dllexport)
#    else
#        define CORE_API __declspec(dllimport)
#    endif
#elif
#    define CORE_API
#endif

#ifdef _MSC_VER
#    define CDECL __cdecl
#else
#    define CDECL __attribute__((__cdecl__))
#endif // _MSC_VER

#ifdef __CUDACC__
#define CUDA_FUNCTION inline __host__ __device__
#else
#define CUDA_FUNCTION inline
#endif // __CUDACC__