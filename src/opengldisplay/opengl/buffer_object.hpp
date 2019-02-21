#pragma once

#include <glad/glad.h>
#include <stdexcept>

namespace opengl {

enum class BufferType : GLenum {
	ARRAY = GL_ARRAY_BUFFER,
	ATOMIC_COUNTER = GL_ATOMIC_COUNTER_BUFFER,
	COPY_READ = GL_COPY_READ_BUFFER,
	COPY_WRITE = GL_COPY_WRITE_BUFFER,
	DISPATCH_INDIRECT = GL_DISPATCH_INDIRECT_BUFFER,
	ELEMENT_ARRAY = GL_ELEMENT_ARRAY_BUFFER,
	PIXEL_PACK = GL_PIXEL_PACK_BUFFER,
	PIXEL_UNPACK = GL_PIXEL_UNPACK_BUFFER,
	QUERY = GL_QUERY_BUFFER,
	SHADER_STORAGE = GL_SHADER_STORAGE_BUFFER,
	TEXTURE = GL_TEXTURE_BUFFER,
	TRANSFORM_FEEDBACK = GL_TRANSFORM_FEEDBACK_BUFFER,
	UNIFORM = GL_UNIFORM_BUFFER
};

class BufferObject {
public:
	BufferObject(BufferType type) :
		m_type(type)
	{
		::glGenBuffers(1u, &m_id);
		if(m_id == 0u)
			throw std::runtime_error("Failed to create buffer object");
	}

	BufferObject(const BufferObject&) = delete;
	BufferObject(BufferObject&& bo) :
		m_id(bo.m_id),
		m_type(bo.m_type)
	{
		bo.m_id = 0u;
	}
	BufferObject& operator=(const BufferObject&) = delete;
	BufferObject& operator=(BufferObject&& bo) {
		std::swap(m_id, bo.m_id);
		std::swap(m_type, bo.m_type);
		return *this;
	}

	~BufferObject() {
		if(m_id != 0)
			::glDeleteBuffers(1u, &m_id);
	}

	void bind() {
		if(m_id == 0u)
			throw std::runtime_error("Invalid vertex array object");
		::glBindBuffer(static_cast<std::underlying_type_t<BufferType>>(m_type), m_id);
	}

	void unbind() {
		::glBindBuffer(static_cast<std::underlying_type_t<BufferType>>(m_type), 0);
	}

private:
	GLuint m_id;
	BufferType m_type;
};

} // namespace opengl