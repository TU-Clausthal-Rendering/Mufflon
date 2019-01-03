#pragma once

#include "core/export/api.h"
#include "util/types.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#endif // _MSC_VER

namespace mufflon { namespace cuda {

// Sync all threads in a block.
CUDA_FUNCTION void syncthreads() {
#ifdef __CUDA_ARCH__
	__syncthreads();
#endif // __CUDA_ARCH__
	// TODO: sync openMP?
}


CUDA_FUNCTION void globalMemoryBarrier() {
#ifdef __CUDA_ARCH__
	__threadfence_system();
#endif // __CUDA_ARCH__
}


// Count the number of consecutive high-order zero bits
CUDA_FUNCTION u64 clz(u64 v) {
#ifdef __CUDA_ARCH__
	return __clzll(v);
#else
#ifdef _MSC_VER
	return __lzcnt64(v);
#else
	return (v == 0) ? 64 : 63 - (u64)log2f((float)v);
#endif // _MSC_VER
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION u32 clz(u32 v) {
#ifdef __CUDA_ARCH__
	return __clz(v);
#else
#ifdef _MSC_VER
	//return __lzcnt(v); //This instruction is carbage: on some machines it is a different opcode -> wrong values
	unsigned long out;
	_BitScanReverse(&out, v);
	return (v == 0) ? 32 : 31 - out;
#else
	return (v == 0) ? 32 : 31 - (u32)log2f((float)v);
#endif // _MSC_VER
#endif // __CUDA_ARCH__
}

}}