#include "gl_wrapper.hpp"
#include "glad/glad.h"
#include <memory>

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

	void clearBufferData(Handle h, size_t clearValueSize, size_t numValues, const void* clearValue) {
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

		if(format) {
			// valid format found => use the fast method
			glClearNamedBufferData(h, internalFormat, format, type, clearValue);
		} else {
			// create temporary buffer
			auto tmp = std::make_unique<char[]>(clearValueSize * numValues);
			// fill with clear value
			for(size_t i = 0; i < numValues; ++i) {
				memcpy(tmp.get() + i * clearValueSize, clearValue, clearValueSize);
			}
			// upload
			bufferSubData(h, 0, clearValueSize * numValues, tmp.get());
		}

	}

	void getBufferSubData(Handle h, size_t offset, size_t size, void* dstData){
		glGetNamedBufferSubData(h, offset, size, dstData);
	}
}
