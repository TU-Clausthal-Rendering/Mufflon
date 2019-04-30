#pragma once

#include "residency.hpp"
#include "util/assert.hpp"
#include "core/cuda/error.hpp"
#include "core/opengl/gl_wrapper.h"
#include <stdexcept>

namespace mufflon { // There is no memory namespace on purpose

namespace memory_details {

void copy_element(const void* element, void* targetMem, const std::size_t elemBytes,
				  const std::size_t count);

} // namespace memory_details

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

	template < class T, bool Init = true, typename... Args >
	static T* alloc_array(std::size_t n, Args... args) {
		// Get the memory
		T* ptr = reinterpret_cast<T*>(new unsigned char[sizeof(T) * n]);
		if(ptr == nullptr)
			throw BadAllocation<DEVICE>();
		// Initialize it
		if(Init)
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

	template < class T, bool Init = true, typename... Args >
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
		if(!std::is_fundamental<T>::value && Init) {
			T prototype{ std::forward<Args>(args)... };
			memory_details::copy_element(&prototype, ptr, sizeof(T), n);
		}
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
		copy(newPtr, ptr, 0, prev);
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
};

template <>
class Allocator<Device::OPENGL> {
public:
	static constexpr Device DEVICE = Device::OPENGL;

	template < class T, typename... Args >
	static gl::Handle alloc(Args... args) {
		return alloc_array<T>(1, std::forward<Args>(args)...);
	}

	template < class T, bool Init = true, typename... Args >
	static gl::Handle alloc_array(std::size_t n, Args... args) {
		static_assert(std::is_trivially_copyable<T>::value,
			"Must be trivially copyable");

		// create handle
		const auto byteSize = n * sizeof(T);
		const auto id = gl::genBuffer();
		gl::bindBuffer(gl::BufferType::ShaderStorage, id);
		gl::bufferStorage(id, byteSize, nullptr, gl::StorageFlags::DynamicStorage);
		// transfer data
		if (Init) {
			T prototype{ std::forward<Args>(args)... };
			gl::clearBufferData(id, n, &prototype);
		}
		
		return id;
	}

	template < class T >
	static gl::Handle realloc(gl::Handle handle, std::size_t prev, std::size_t next) {
		// create new buffer and copy contents of old buffer
		mAssert(next > prev);

		const auto newHandle = gl::genBuffer();
		gl::bindBuffer(gl::BufferType::ShaderStorage, newHandle);
		gl::bufferStorage(newHandle, next, nullptr, gl::StorageFlags::DynamicStorage);
		// transfer data
		gl::copyBufferSubData(handle, newHandle, 0, 0, prev);
		// cleanup
		gl::deleteBuffer(handle);
		return newHandle;
	}

	template < class T >
	static gl::Handle free(gl::Handle handle, std::size_t n) {
		if(handle != 0)
			gl::deleteBuffer(handle);
		return 0;
	}
};

// Deleter for the above custom allocated memories.
template < Device dev >
class Deleter {
public:
	Deleter() :				 m_n(0) {}
	Deleter(std::size_t n) : m_n(n) {}

	std::size_t get_size() const noexcept { return m_n; }

	template < typename T >
	//void operator () (ArrayDevHandle_t<dev, T> p) const {
	void operator () (T* p) const {
		Allocator<dev>::template free<T>(p, m_n);
	}
private:
	std::size_t m_n;
};

template <>
class Deleter<Device::OPENGL> {
public:
	Deleter() : m_n(0) {}
	Deleter(std::size_t n) : m_n(n) {}

	std::size_t get_size() const noexcept { return m_n; }

	template < typename T >
	void operator () (ArrayDevHandle_t<Device::OPENGL, T> p) const {
		Allocator<Device::OPENGL>::template free<T>(p, m_n);
	}
private:
	std::size_t m_n;
};
}
// TODO remove include from this place
#include "unique_device_ptr.h"