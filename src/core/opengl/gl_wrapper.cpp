#include "gl_wrapper.h"
#include "glad/glad.h"

namespace mufflon::gl {

	Handle genBuffer() {
		Handle res;
		glGenBuffers(1, &res);
		return res;
	}

	void bindBuffer(BufferType target, Handle id) {
		glBindBuffer(GLenum(target), id);
	}

	void bufferStorage(Handle id, size_t size, const void* data, StorageFlags flags) {
		glNamedBufferStorage(id, GLsizeiptr(size), data, GLbitfield(flags));
	}

	void copyBufferSubData(Handle src, Handle dst, size_t srcOffset, size_t dstOffset, size_t size)	{
		glCopyNamedBufferSubData(src, dst, srcOffset, dstOffset, size);
	}

	void deleteBuffer(Handle h) {
		glDeleteBuffers(1, &h); 
	}

	void bufferSubData(Handle h, size_t offset, size_t size, const void* data) {
		glNamedBufferSubData(h, offset, size, data);
	}
}
