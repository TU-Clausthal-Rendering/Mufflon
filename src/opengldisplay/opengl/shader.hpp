#pragma once

#include <glad/glad.h>
#include <stdexcept>
#include <utility>

namespace opengl {


class Shader {
public:
	enum class Type : GLenum {
		VERTEX_SHADER = GL_VERTEX_SHADER,
		FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
		GEOMETRY_SHADER = GL_GEOMETRY_SHADER
	};

	enum class Parameter : GLenum {
		SHADER_TYPE = GL_SHADER_TYPE,
		DELETE_STATUS = GL_DELETE_STATUS,
		COMPILE_STATUS = GL_COMPILE_STATUS,
		INFO_LOG_LENGTH = GL_INFO_LOG_LENGTH,
		SHADER_SOURCE_LENGTH = GL_SHADER_SOURCE_LENGTH
	};

	Shader(Type type) : m_id(::glCreateShader(static_cast<GLenum>(type))), m_type(type) {
		if(m_id == 0u)
			throw std::runtime_error("Failed to create shader object");
	}
	Shader(const Shader&) = delete;
	Shader(Shader&& shader) : m_id(shader.m_id), m_type(shader.m_type) {
		shader.m_id = 0u;
	}
	Shader& operator=(const Shader&) = delete;
	Shader& operator=(Shader&& shader) {
		std::swap(m_id, shader.m_id);
		m_type = shader.m_type;
		return *this;
	}
	~Shader() {
		if(m_id != 0u)
			::glDeleteShader(m_id);
	}

	GLint get_parameter(Parameter param) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid shader object");
		GLint result = 0;
		::glGetShaderiv(m_id, static_cast<GLenum>(param), &result);
		return result;
	}

	void attach_source(const char* code) {
		if(m_id == 0u)
			throw std::runtime_error("Invalid shader object");
		GLint length = static_cast<GLint>(std::strlen(code));
		::glShaderSource(m_id, 1u, &code, &length);
	}

	bool compile() {
		if(m_id == 0u)
			throw std::runtime_error("Invalid shader object");
		::glCompileShader(m_id);
		if(get_parameter(Parameter::COMPILE_STATUS) != GL_TRUE)
		   return false;
		return true;
	}

	Type get_type() const noexcept {
		return m_type;
	}

	std::string get_info_log() const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid shader object");
		GLint logLength = get_parameter(Parameter::INFO_LOG_LENGTH);
		std::string msg;
		msg.resize(logLength + 1);
		::glGetShaderInfoLog(m_id, logLength, NULL, msg.data());
		return msg;
	}

	std::string get_shader_source() const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid shader object");
		std::string code;
		code.resize(get_parameter(Parameter::SHADER_SOURCE_LENGTH));
		GLsizei length;
		::glGetShaderSource(m_id, static_cast<GLsizei>(code.size()), &length, code.data());
		if(length != code.size())
			code.resize(length);
		return code;
	}

	void attach_to(GLuint programId) {
		if(programId == 0u)
			throw std::runtime_error("Invalid program ID");
		::glAttachShader(programId, m_id);
	}

	void detach_from(GLuint programId) {
		if(programId == 0u)
			throw std::runtime_error("Invalid program ID");
		::glDetachShader(programId, m_id);
	}

private:
	GLuint m_id;
	Type m_type;
};

} // namespace opengl