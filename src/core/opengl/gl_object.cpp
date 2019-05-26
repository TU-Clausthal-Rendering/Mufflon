#include "gl_object.h"
#include <glad/glad.h>

void mufflon::gl::detail::TextureDeleter::del(gl::Handle handle) noexcept {
	glDeleteTextures(1, &handle);
}

void mufflon::gl::detail::BufferDeleter::del(gl::Handle handle) noexcept {
	glDeleteBuffers(1, &handle);
}

void mufflon::gl::detail::FramebufferDeleter::del(gl::Handle handle) noexcept {
	glDeleteFramebuffers(1, &handle);
}

void mufflon::gl::detail::ShaderDeleter::del(gl::Handle handle) noexcept {
	glDeleteShader(handle);
}

void mufflon::gl::detail::ProgramDeleter::del(gl::Handle handle) noexcept {
	glDeleteProgram(handle);
}

void mufflon::gl::detail::VertexArrayDeleter::del(gl::Handle handle) noexcept {
	glDeleteVertexArrays(1, &handle);
}

void mufflon::gl::detail::SamplerDeleter::del(gl::Handle handle) noexcept {
	glDeleteSamplers(1, &handle);
}
