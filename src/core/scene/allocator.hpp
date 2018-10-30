#pragma once

#include "residency.hpp"
#include "core/cuda/error.hpp"
#include <stdexcept>

namespace mufflon::scene {

// Error class for per-device allocation failure

template < Device dev >
class BadAllocation : public std::exception {
public:
	static constexpr Device DEVICE = dev;
};

// Allocator class providing alloc/realloc/free for attribute allocation
template < Device dev >
class Allocator;

// Allocator specialization for CPU
template <>
class Allocator<Device::CPU> {
public:
	static constexpr Device DEVICE = Device::CPU;

	template < class T >
	static T* alloc(std::size_t n) {
		T* ptr = new T[n];
		if(ptr == nullptr)
			throw BadAllocation<DEVICE>();
		return ptr;
	}

	template < class T >
	static T* realloc(T* ptr, std::size_t prev, std::size_t next) {
		(void)prev;
		void* newPtr = std::realloc(ptr, sizeof(T) * next);
		if(newPtr == nullptr)
			throw BadAllocation<DEVICE>();
		return reinterpret_cast<T*>(newPtr);
	}

	template < class T >
	static void free(T* ptr, std::size_t n) {
		delete[] ptr;
	}

	template < class T >
	static void copy(T* dst, const T* src, std::size_t n) {
		std::memcpy(dst, src, sizeof(T) * n);
	}
};

// Allocator specialization for CUDA
template <>
class Allocator<Device::CUDA> {
public:
	static constexpr Device DEVICE = Device::CUDA;

	template < class T >
	static T* alloc(std::size_t n) {
		T* ptr = nullptr;
		cuda::check_error(cudaMalloc(&ptr, sizeof(T) * n));
		if(ptr == nullptr)
			throw BadAllocation<DEVICE>();
		return ptr;
	}

	template < class T >
	static T* realloc(T* ptr, std::size_t prev, std::size_t next) {
		T* newPtr = alloc<T>(next);
		copy(newPtr, ptr, prev);
		free(ptr, prev);
		return newPtr;
	}

	template < class T >
	static void free(T* ptr, std::size_t n) {
		cuda::check_error(cudaFree(ptr));
	}

	template < class T >
	static void copy(T* dst, const T* src, std::size_t n) {
		cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice);
	}
};


} // namespace mufflon::util