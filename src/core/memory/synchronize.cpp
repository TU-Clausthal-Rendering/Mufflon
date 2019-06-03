#include "synchronize.hpp"
#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace mufflon {

void copy_cuda_opengl(void* dst, GLuint src, std::size_t byteOffset, std::size_t size) {
	// Gotta map the OpenGL resource
	// TODO: maybe we can find out if we need to maintain parts of the buffer or not? For now assume we do
	// TODO: map buffers anyway?
	cudaGraphicsResource_t resource;
	void* mappedPtr = nullptr;
	size_t mappedSize;
	cuda::check_error(cudaGraphicsGLRegisterBuffer(&resource, src, cudaGraphicsRegisterFlagsReadOnly));
	cuda::check_error(cudaGraphicsMapResources(1, &resource));
	cuda::check_error(cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, resource));
	cuda::check_error(cudaMemcpy(dst, static_cast<const char*>(mappedPtr) + byteOffset, size, cudaMemcpyDefault));
	cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
	cuda::check_error(cudaGraphicsUnregisterResource(resource));
}

void copy_cuda_opengl(GLuint dst, std::size_t byteOffset, const void* src, std::size_t size) {
	// Gotta map the OpenGL resource
	// TODO: maybe we can find out if we need to maintain parts of the buffer or not? For now assume we do
	// TODO: map buffers anyway?
	cudaGraphicsResource_t resource;
	void* mappedPtr = nullptr;
	size_t mappedSize;
	cuda::check_error(cudaGraphicsGLRegisterBuffer(&resource, dst, cudaGraphicsRegisterFlagsNone));
	cuda::check_error(cudaGraphicsMapResources(1, &resource));
	cuda::check_error(cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, resource));
	cuda::check_error(cudaMemcpy(static_cast<char*>(mappedPtr) + byteOffset, src, size, cudaMemcpyDefault));
	cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
	cuda::check_error(cudaGraphicsUnregisterResource(resource));
}

} // namespace mufflon