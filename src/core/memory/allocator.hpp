#pragma once

#include "residency.hpp"
#include "core/cuda/error.hpp"
#include <stdexcept>

namespace mufflon { // There is no memory namespace on purpose

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

	template < class T, typename... Args >
	static T* alloc(Args... args) {
		return alloc_array<T>(1, std::forward<Args>(args)...);
	}

	template < class T, typename... Args >
	static T* alloc_array(std::size_t n, Args... args) {
		// Get the memory
		T* ptr = reinterpret_cast<T*>(new unsigned char[sizeof(T) * n]);
		if(ptr == nullptr)
			throw BadAllocation<DEVICE>();
		// Initialize it
		for(std::size_t i = 0; i < n; ++i)
			new (ptr+i) T {std::forward<Args>(args)...};
return ptr;
	}

	// Danger: realloc does not handle construction/destruction
	template < class T >
	static T* realloc(T* ptr, std::size_t prev, std::size_t next) {
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		static_assert(std::is_trivially_constructible<T>::value,
					  "Must be trivially constructible");
		static_assert(std::is_trivially_destructible<T>::value,
					  "Must be trivially destructible");
		(void)prev;
		void* newPtr = std::realloc(ptr, sizeof(T) * next);
		if(newPtr == nullptr)
			throw BadAllocation<DEVICE>();
		return reinterpret_cast<T*>(newPtr);
	}

	template < class T >
	static T* free(T* ptr, std::size_t n) {
		if(ptr != nullptr) {
			// Call destructors manually, because the memory was allocated raw
			for(std::size_t i = 0; i < n; ++i)
				ptr[i].~T();
			delete[](unsigned char*)ptr;
		}
		return nullptr;
	}

	template < class T >
	static void copy(T* dst, const T* src, std::size_t n) {
		std::memcpy(dst, src, sizeof(T) * n);
	}

	template < class T >
	static void copy_cuda(T* dst, const T* src, std::size_t n) {
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		cuda::check_error(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
	}
};

// Allocator specialization for CUDA
template <>
class Allocator<Device::CUDA> {
public:
	static constexpr Device DEVICE = Device::CUDA;

	template < class T, typename... Args >
	static T* alloc(Args... args) {
		return alloc_array<T>(1, std::forward<Args>(args)...);
	}

	template < class T, typename... Args >
	static T* alloc_array(std::size_t n, Args... args) {
		// Get the memory
		T* ptr = nullptr;
		cuda::check_error(cudaMalloc(&ptr, sizeof(T) * n));
		if(ptr == nullptr)
			throw BadAllocation<DEVICE>();
		// TODO: initialize only if necessary? But when is this.
		// - not for elementary types like char, int, ... except there is an 'args'?
		// Initialize it
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		T prototype{ std::forward<Args>(args)... };
		for(std::size_t i = 0; i < n; ++i)
			cuda::check_error(cudaMemcpy(ptr + i, &prototype, sizeof(T), cudaMemcpyHostToDevice));
		return ptr;
	}

	// Danger: realloc does not handle construction/destruction
	template < class T >
	static T* realloc(T* ptr, std::size_t prev, std::size_t next) {
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		static_assert(std::is_trivially_constructible<T>::value,
					  "Must be trivially constructible");
		static_assert(std::is_trivially_destructible<T>::value,
					  "Must be trivially destructible");
		T* newPtr = alloc_array<T>(next);
		copy(newPtr, ptr, prev);
		free(ptr, prev);
		return newPtr;
	}

	template < class T >
	static T* free(T* ptr, std::size_t n) {
		if(ptr != nullptr) {
			// Call destructors manually, because the memory was allocated raw
			for(std::size_t i = 0; i < n; ++i)
				ptr[i].~T();
			cuda::check_error(cudaFree(ptr));
		}
		return nullptr;
	}

	template < class T >
	static void copy(T* dst, const T* src, std::size_t n) {
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		cuda::check_error(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice));
	}

	template < class T >
	static void copy_cpu(T* dst, const T* src, std::size_t n) {
		static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
		cuda::check_error(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
	}
};

// Deleter for the above custom allocated memories.
template < Device dev >
class Deleter {
public:
	Deleter() :				 m_n(0) {}
	Deleter(std::size_t n) : m_n(n) {}

	template < typename T >
	void operator () (T * p) const {
		Allocator<dev>::template free<T>(p, m_n);
	}
private:
	std::size_t m_n;
};

// Helper alias to simplyfy the construction of managed (unique_ptr) memory with the
// custom allocator.
template < Device dev, typename T >
using unique_device_ptr = std::unique_ptr<T, Deleter<dev>>;

template < Device dev, typename T, typename... Args > inline unique_device_ptr<dev,T>
make_udevptr(Args... args) {
	return unique_device_ptr<dev,T>(
		Allocator<dev>::template alloc<T>(std::forward<Args>(args)...),
		Deleter<dev>(1)
	);
}

template < Device dev, typename T, typename... Args > inline unique_device_ptr<dev,T>
make_udevptr_array(std::size_t n, Args... args) {
	return unique_device_ptr<dev,T>(
		Allocator<dev>::template alloc_array<std::remove_pointer_t<std::decay_t<T>>>(n, std::forward<Args>(args)...),
		Deleter<dev>(n)
	);
}


} // namespace mufflon::util