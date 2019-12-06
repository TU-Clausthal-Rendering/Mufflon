#pragma once

#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "util/types.hpp"
#include <atomic>

#ifdef _MSC_VER
#include <intrin.h>
#endif // _MSC_VER

namespace mufflon { namespace cuda {

namespace atomic_details {

// Atomic values: on CUDA side they're just regular values, while C++ expects std::atomic for atomic operations
template < Device dev, class T >
struct AtomicValue { using Type = T; };
template < class T >
struct AtomicValue<Device::CPU, T> { using Type = std::atomic<T>; };
// TODO GL
template < class T > 
struct AtomicValue<Device::OPENGL, T> { using Type = std::atomic<T>; };

// Implementations of atomic-exchange
template < Device dev, class T >
struct AtomicOps {};
template <  class T >
struct AtomicOps<Device::CPU, T> {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static T exchange(typename AtomicValue<Device::CPU, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		return std::atomic_exchange_explicit(&atom, value, std::memory_order::memory_order_relaxed);
	}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static void add(typename AtomicValue<Device::CPU, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		T expected = atom.load();
		T desired;
		do {
			desired = expected + value;
		} while(!atom.compare_exchange_weak(expected, desired));
	}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static T load(const typename AtomicValue<Device::CPU, T>::Type& atom) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		return atom.load();
	}
};

template <>
struct AtomicOps<Device::CPU, i32> {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static i32 exchange(typename AtomicValue<Device::CPU, i32>::Type& atom, const i32 value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		return std::atomic_exchange_explicit(&atom, value, std::memory_order::memory_order_relaxed);
}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static void add(typename AtomicValue<Device::CPU, i32>::Type& atom, const i32 value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		atom += value;
	}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static i32 load(const typename AtomicValue<Device::CPU, i32>::Type& atom) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		return atom.load();
	}
};

template <  class T >
struct AtomicOps<Device::CUDA, T> {
	__host__ __device__ static T exchange(typename AtomicValue<Device::CUDA, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		return atomicExch(&atom, value);
#else // __CUDA_ARCH__
		return T{};
#endif // __CUDA_ARCH__
	}

	__host__ __device__ static void add(typename AtomicValue<Device::CUDA, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		(void)atomicAdd(&atom, value);
#endif // __CUDA_ARCH__
	}

	__host__ __device__ static T load(const typename AtomicValue<Device::CUDA, T>::Type& atom) {
#ifdef __CUDA_ARCH__
		return atom;
#else // __CUDA_ARCH__
		return T{};
#endif // __CUDA_ARCH__
	}
};

// TODO GL
template <  class T >
struct AtomicOps<Device::OPENGL, T> {
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static T exchange(typename AtomicValue<Device::OPENGL, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		return std::atomic_exchange_explicit(&atom, value, std::memory_order::memory_order_relaxed);
	}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // __CUDACC__
	__host__ __device__ static void add(typename AtomicValue<Device::OPENGL, T>::Type& atom, const T value) {
#ifdef __CUDA_ARCH__
		mAssertMsg(false, "This function must not be called on the GPU!");
#endif // __CUDA_ARCH__
		T expected = atom.load();
		T desired;
		do {
			desired = expected + value;
		} while (!atom.compare_exchange_weak(expected, desired));
	}
};

} // namespace atomic_details

template < Device dev, class T >
using Atomic = typename atomic_details::AtomicValue<dev, T>::Type;

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
#else // __CUDA_ARCH__
	// TODO: strongest guarantee, but does it carry the same meaning as CUDAs equivalent?
	std::atomic_thread_fence(std::memory_order::memory_order_seq_cst);
#endif // __CUDA_ARCH__
}


// Count the number of consecutive high-order zero bits
CUDA_FUNCTION u64 clz(u64 v) {
#ifdef __CUDA_ARCH__
	return __clzll(v);
#else
#ifdef _MSC_VER
	// return __lzcnt64(v); //This instruction is carbage: on some machines it is a different opcode -> wrong values
	unsigned long out;
	_BitScanReverse64(&out, v);
	return (v == 0) ? 64 : 63 - out;
#else
	// TODO: This will not result in the correct value, since float can represent MUCH less values than 64 bit ints!
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

// Atomic operations; gives no(!) guarantees about memory access ordering
// Note: this function must not be called on the GPU for CPU atomics!
template < Device dev, class T >
CUDA_FUNCTION T atomic_exchange(Atomic<dev, T>& atom, const T value) {
	return atomic_details::AtomicOps<dev, T>::exchange(atom, value);
}

// Note: this function must not be called on the GPU for CPU atomics!
template < Device dev, class T >
CUDA_FUNCTION void atomic_add(Atomic<dev, T>& atom, const T value) {
	atomic_details::AtomicOps<dev, T>::add(atom, value);
}

// Note: this function must not be called on the GPU for CPU atomics!
template < Device dev, class T >
CUDA_FUNCTION T atomic_load(const Atomic<dev, T>& atom) {
	return atomic_details::AtomicOps<dev, T>::load(atom);
}

}} // namespace mufflon::cuda