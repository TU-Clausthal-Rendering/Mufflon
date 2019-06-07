#pragma once

#include "residency.hpp"
#include "core/cuda/error.hpp"
#include "core/opengl/gl_texture.hpp"
#include "core/opengl/gl_buffer.hpp"

namespace mufflon { // There is no memory namespace on purpose


// Functions for unloading a handle from the device
template < class T >
void unload(ArrayDevHandle_t<Device::CPU, T>& hdl) {
	delete[] hdl.handle;
	hdl.handle = nullptr;
}
template < class T >
void unload(ArrayDevHandle_t<Device::CUDA, T>& hdl) {
	if(hdl.handle != nullptr) {
		cuda::check_error(cudaFree(hdl.handle));
		hdl.handle = nullptr;
	}
}
template < class T >
void unload(ArrayDevHandle_t<Device::OPENGL, T>& hdl) {
	gl::deleteBuffer(hdl.id);
}

// A number of copy primitives which call the internal required methods.
// This relies on CUDA UVA
template < typename T >
inline void copy(T* dst, const T* src, std::size_t size ) {
	static_assert(std::is_trivially_copyable<T>::value,
					  "Must be trivially copyable");
	cuda::check_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}

// Copy operations between OpenGL and CUDA
void copy_cuda_opengl(u32 dst, std::size_t byteOffset, const void* src, std::size_t size);
void copy_cuda_opengl(void* dst, u32 src, std::size_t byteOffset, std::size_t size);

template < typename T >
inline void copy(gl::BufferHandle<T> dst, const T* src, std::size_t size) {
	static_assert(std::is_trivially_copyable<T>::value,
		"Must be trivially copyable");
	// Gotta check where the pointer resides
	cudaPointerAttributes attribs;
	auto res = cudaPointerGetAttributes(&attribs, src);
	if(res == cudaErrorInvalidValue || attribs.type == cudaMemoryTypeUnregistered  || attribs.type == cudaMemoryTypeHost) {
		// Host memory; gotta clear the CUDA error
		gl::bufferSubData(dst.id, dst.get_byte_offset(), size, src);
		cudaGetLastError();
	} else {
		copy_cuda_opengl(dst.id, dst.offset * sizeof(T), static_cast<const void*>(src), size);
	}
}

template < typename T >
inline void copy(T* dst, gl::BufferHandle<T> src, std::size_t size) {
	// Gotta check where the pointer resides
	cudaPointerAttributes attribs;
	auto res = cudaPointerGetAttributes(&attribs, dst);
	if(res == cudaErrorInvalidValue || attribs.type == cudaMemoryTypeUnregistered || attribs.type == cudaMemoryTypeHost) {
		// Host memory; gotta clear the CUDA error
		gl::getBufferSubData(src.id, src.get_byte_offset(), size, dst);
		cudaGetLastError();
	} else {
		copy_cuda_opengl(static_cast<void*>(dst), src.id, src.offset * sizeof(T), size);
	}

}

template < typename T >
inline void copy(gl::BufferHandle<T> dst, gl::BufferHandle<T> src, std::size_t size) {
	gl::copyBufferSubData(src.id, dst.id, src.get_byte_offset(), dst.get_byte_offset(), size);
}

template < typename T >
inline void copy(gl::TextureHandle dst, gl::BufferHandle<T> src, std::size_t size) {
	// TODO gl
}

template < Device dev >
inline std::enable_if_t<dev != Device::OPENGL, void> mem_set(void* mem, int value, std::size_t size) {
	std::memset(mem, value, size);
}

template <>
inline void mem_set<Device::CUDA>(void* mem, int value, std::size_t size) {
	cuda::check_error(::cudaMemset(mem, value, size));
}

template < Device dev, class T >
inline std::enable_if_t<dev == Device::OPENGL, void> mem_set(gl::BufferHandle<T> mem, int value, std::size_t size) {
	gl::clearBufferSubData(mem.id, mem.get_byte_offset(), size, value);
}


} // namespace mufflon