#pragma once

#include "shader.hpp"
#include <glad/glad.h>
#include <vector>

namespace opengl {

class Program {
public:
	enum class Parameter : GLenum {
		DELETE_STATUS = GL_DELETE_STATUS,
		LINK_STATUS = GL_LINK_STATUS,
		VALIDATE_STATUS = GL_VALIDATE_STATUS,
		INFO_LOG_LENGTH = GL_INFO_LOG_LENGTH,
		ATTACHED_SHADERS = GL_ATTACHED_SHADERS,
		ACTIVE_ATOMIC_COUNTER_BUFFERS = GL_ACTIVE_ATOMIC_COUNTER_BUFFERS,
		ACTIVE_ATTRIBUTES = GL_ACTIVE_ATTRIBUTES,
		ACTIVE_ATTRIBUTE_MAX_LENGTH = GL_ACTIVE_ATTRIBUTE_MAX_LENGTH,
		ACTIVE_UNIFORMS = GL_ACTIVE_UNIFORMS,
		ACTIVE_UNIFORM_BLOCKS = GL_ACTIVE_UNIFORM_BLOCKS,
		ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH = GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH,
		ACTIVE_UNIFORM_MAX_LENGTH = GL_ACTIVE_UNIFORM_MAX_LENGTH,
		COMPUTE_WORK_GROUP_SIZE = GL_COMPUTE_WORK_GROUP_SIZE,
		PROGRAM_BINARY_LENGTH = GL_PROGRAM_BINARY_LENGTH,
		TRANSFORM_FEEDBACK_BUFFER_MODE = GL_TRANSFORM_FEEDBACK_BUFFER_MODE,
		TRANSFORM_FEEDBACK_VARYINGS = GL_TRANSFORM_FEEDBACK_VARYINGS,
		TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH = GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH,
		GEOMETRY_VERTICES_OUT = GL_GEOMETRY_VERTICES_OUT,
		GEOMETRY_INPUT_TYPE = GL_GEOMETRY_INPUT_TYPE,
		GEOMETRY_OUTPUT_TYPE = GL_GEOMETRY_OUTPUT_TYPE
	};

	Program() : m_id(::glCreateProgram()), m_shaders() {
		if(m_id == 0u)
			throw std::runtime_error("Failed to create program object");
	}
	Program(const Program&) = delete;
	Program(Program&& program) : m_id(program.m_id), m_shaders(std::move(program.m_shaders)) {
		program.m_id = 0u;
	}
	Program& operator=(const Program&) = delete;
	Program& operator=(Program&& program) {
		std::swap(m_id, program.m_id);
		std::swap(m_shaders, program.m_shaders);
		return *this;
	}
	~Program() {
		if(m_id != 0u) {
			if(get_parameter(Parameter::ATTACHED_SHADERS) > 0) {
				// Detach attached shaders that weren't linked yet
				for(Shader& shader : m_shaders)
					shader.detach_from(m_id);

			}
			::glDeleteProgram(m_id);
		}
	}

	std::size_t get_attached_shader_count() const noexcept {
		return m_shaders.size();
	}

	void attach(Shader&& shader) {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		m_shaders.push_back(std::move(shader));
		m_shaders.back().attach_to(m_id);
	}

	void detach_all() {
		for(Shader& shader : m_shaders)
			shader.detach_from(m_id);
		m_shaders.clear();
	}

	bool link() {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		::glLinkProgram(m_id);
		if(get_parameter(Parameter::LINK_STATUS) != GL_TRUE)
			return false;
		for(Shader& shader : m_shaders)
			shader.detach_from(m_id);
		return true;
	}

	GLint get_parameter(Parameter param) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLint result = 0;
		::glGetProgramiv(m_id, static_cast<GLenum>(param), &result);
		return result;
	}

	std::string get_info_log() {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLint logLength = get_parameter(Parameter::INFO_LOG_LENGTH);
		std::string msg;
		msg.resize(logLength + 1);
		::glGetProgramInfoLog(m_id, logLength, NULL, msg.data());
		return msg;
	}

	void activate() {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		::glUseProgram(m_id);
	}

	static void deactivate() {
		::glUseProgram(0u);
	}

	GLuint get_uniform_location(const char* name) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		return ::glGetUniformLocation(m_id, &name[0u]);
	}

	GLuint get_attribute_location(const char* name) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		return ::glGetAttribLocation(m_id, name);
	}

	GLfloat get_uniform_float(GLuint location) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLfloat val;
		::glGetUniformfv(m_id, location, &val);
		return val;
	}

	GLdouble get_uniform_double(GLuint location) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLdouble val;
		::glGetUniformdv(m_id, location, &val);
		return val;
	}

	GLint get_uniform_int(GLuint location) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLint val;
		::glGetUniformiv(m_id, location, &val);
		return val;
	}

	GLuint get_uniform_uint(GLuint location) const {
		if(m_id == 0u)
			throw std::runtime_error("Invalid program object");
		GLuint val;
		::glGetUniformuiv(m_id, location, &val);
		return val;
	}

private:
	GLuint m_id;
	std::vector<Shader> m_shaders;
};

} // namespace opengl