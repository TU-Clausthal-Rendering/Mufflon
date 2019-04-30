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

	void clearBufferData(Handle h, size_t clearValueSize, const void* clearValue) {
		GLenum format = 0;
		GLenum internalFormat = 0;
		GLenum type = 0;
		switch (clearValueSize)
		{
		case 1:
			internalFormat = GL_R8I;
			type = GL_BYTE;
			format = GL_RED;
			break;
		case 2:
			internalFormat = GL_R16I;
			type = GL_SHORT;
			format = GL_RED;
			break;
		case 3:
			internalFormat = GL_RGB8I;
			type = GL_BYTE;
			format = GL_RGB;
			break;
		case 4:
			internalFormat = GL_R32I;
			type = GL_INT;
			format = GL_RED;
			break;
		case 6:
			internalFormat = GL_RGB16I;
			type = GL_SHORT;
			format = GL_RGB;
			break;
		case 8:
			internalFormat = GL_RG32I;
			type = GL_INT;
			format = GL_RG;
			break;
		case 12:
			internalFormat = GL_RGB32I;
			type = GL_INT;
			format = GL_RGB;
			break;
		case 16:
			internalFormat = GL_RGBA32I;
			type = GL_INT;
			format = GL_RGBA;
			break;
		}

		glClearNamedBufferData(h, internalFormat, format, type, clearValue);
	}

	void getBufferSubData(Handle h, size_t offset, size_t size, void* dstData){
		glGetNamedBufferSubData(h, offset, size, dstData);
	}
}
