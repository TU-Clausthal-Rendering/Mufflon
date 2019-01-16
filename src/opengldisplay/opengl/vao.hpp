#pragma once

#include <glad/glad.h>
#include <stdexcept>
#include <utility>

namespace opengl {

// We don't need the actual functionality of the VAO, we just need one bound
class VertexArray {
public:
	VertexArray() :
		m_id(0)
	{
		::glGenVertexArrays(1u, &m_id);
		if(m_id == 0u)
			throw std::runtime_error("Failed to create vertex array object");
	}
	VertexArray(const VertexArray&) = delete;
	VertexArray(VertexArray&& vao) : m_id(vao.m_id) {
		vao.m_id = 0u;
	}
	VertexArray& operator=(const VertexArray&) = delete;
	VertexArray& operator=(VertexArray&& vao) {
		std::swap(m_id, vao.m_id);
		return *this;
	}
	~VertexArray() {
		if(m_id != 0u)
			::glDeleteVertexArrays(1u, &m_id);
	}

	void bind() const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid vertex array object");
		::glBindVertexArray(m_id);
	}

	static void unbind() {
		::glBindVertexArray(0u);
	}

private:
	GLuint m_id;
};

} // namespace opengl