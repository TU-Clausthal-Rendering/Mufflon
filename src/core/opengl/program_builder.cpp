#include "util/log.hpp"
#include "program_builder.h"
#include <glad/glad.h>
#include <fstream>
#include <sstream>

namespace mufflon::gl {

    ProgramBuilder::~ProgramBuilder() {
		unload(0);
    }

    ProgramBuilder& ProgramBuilder::add_file(ShaderType type, const std::string& filename) {
		std::ifstream file;
		file.open(filename.c_str());

        
		if(!file.is_open()) {
			logError("could not open shader file: ", filename);
		}

		// string stream to convert file into string
		std::stringstream sstream;
		sstream << file.rdbuf();
		file.close();

		return add_source(type, sstream.str(), filename);
    }

    ProgramBuilder& ProgramBuilder::add_source(ShaderType type, const std::string& source, const std::string& debugName) {
		const gl::Handle shader = glCreateShader(GLenum(type));
		m_attachments.push_back(shader);

        auto src = source.c_str();
		glShaderSource(shader, 1, &src, nullptr);
		
        glCompileShader(shader);
		GLint isCompiled = GL_FALSE;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

		if(isCompiled) return *this;

		// Read out the error message.
		GLint length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		std::string errorLog;
		errorLog.resize(length);
		glGetShaderInfoLog(shader, length, &length, &errorLog[0]);

		logError("failed to compile shader ", debugName, " ", errorLog);

		return *this;
    }

    gl::Handle ProgramBuilder::build() {
		const gl::Handle id = glCreateProgram();
		
        for(auto a : m_attachments)
			glAttachShader(id, a);

		glLinkProgram(id);

        // free shader attachments
		unload(id);

		GLint isLinked = GL_FALSE;
		glGetProgramiv(id, GL_LINK_STATUS, &isLinked);

        if(!isLinked) {
			GLint length = 0;
			glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);
			std::string errorLog;
			errorLog.resize(length);
			glGetProgramInfoLog(id, length, &length, &errorLog[0]);

			logError("failed to link program ", errorLog);
        }

		return id;
    }

    void ProgramBuilder::unload(gl::Handle program) {
		for(auto a : m_attachments) {
            if(program)
			    glDetachShader(program, a);
			glDeleteShader(a);
		}
		m_attachments.clear();
    }
}
