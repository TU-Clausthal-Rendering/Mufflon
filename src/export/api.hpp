#pragma once

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#    define CDECL __cdecl
#elif
#    define LIBRARY_API
#    define CDECL __attribute__((__cdecl__))
#endif

#ifdef __CUDACC__
#define CUDA_FUNCTION inline __host__ __device__
#else
#define CUDA_FUNCTION inline
#endif // __CUDACC__