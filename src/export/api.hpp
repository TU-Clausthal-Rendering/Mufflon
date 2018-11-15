#pragma once

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elif
#    define LIBRARY_API
#endif

#ifdef __CUDACC__
#define CUDA_FUNCTION inline __host__ __device__
#else
#define CUDA_FUNCTION inline
#endif // __CUDACC__