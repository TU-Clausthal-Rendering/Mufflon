#include "util/log.hpp"
#include "program_builder.h"
#include <glad/glad.h>
#include <fstream>
#include <sstream>
#include <functional>
#include <regex>

namespace mufflon::gl {

	// converts opengl file ids to custom strings
	// log: the log returned by getInfoLog() or thrown by compile()
	// convertFunction: function that converts opengl file numbers to strings
	// returns log with file numbers replaced by strings from the converter function
	static std::string convertLog(const std::string& log, const std::function<std::string(GLint)>& convertFunction) {
		// convert error information with used files table
		// errors are like 5(20): => error in file 5 line 20
		//const std::regex expr("\n[0-9][0-9]*\\([0-9][0-9]*\\):");
		const std::regex expr("[0-9][0-9]*\\([1-9][0-9]*\\)");
		std::smatch m;

		std::string error;
		std::string remaining = log;

		while(std::regex_search(remaining, m, expr))
		{
			error += m.prefix();

			// append the correct filename
			// extract number
			const auto parOpen = m.str().find('(');
			const auto fileNumber = m.str().substr(0, parOpen);

			const auto num = std::stoi(fileNumber);
			error += convertFunction(GLint(num));
			error += m.str().substr(parOpen);

			remaining = m.suffix().str();
		}

		error += remaining;

		return error;
	}

    ProgramBuilder::ProgramBuilder(std::string version) :
        m_version(std::move(version)){
    }

    ProgramBuilder& ProgramBuilder::add_file(const std::string& filename, bool isCommon) {
		static std::unordered_map<std::string, std::string> s_cachedFiles;

        // was the file already loaded?
		const auto it = s_cachedFiles.find(filename);
        if(it != s_cachedFiles.end()) {
			return add_source(it->second, "file: "+ filename, isCommon);
        }

        // load file
		std::ifstream file;
		file.open(filename.c_str());
        
		if(!file.is_open()) {
			logError("could not open shader file: ", filename);
		}

		// string stream to convert file into string
		std::stringstream sstream;
		sstream << file.rdbuf();
		file.close();

        // add to cache
		s_cachedFiles[filename] = sstream.str();

		return add_source(sstream.str(), "file: " + filename, isCommon);
    }

    ProgramBuilder& ProgramBuilder::add_source(std::string source, std::string debugName, bool isCommon) {
		add_include(IncludeFile{ std::move(debugName), std::move(source) }, isCommon);
		return *this;
    }

    ProgramBuilder& ProgramBuilder::add_define(std::string name, bool isCommon) {
        add_include(IncludeFile{
            "definition: " + name,
            "#define " + name
			}, isCommon);
		return *this;
    }

    ProgramBuilder& ProgramBuilder::build_shader(ShaderType type) {
		const gl::Handle shader = glCreateShader(GLenum(type));
        m_attachments.emplace_back(gl::Shader(shader));

		// combine sources
		std::stringstream source;
		int curFile = 0;
		source << "#version " << m_version << '\n';
        // use bindless extension
		source << "#extension GL_ARB_bindless_texture : require\n";
		for(auto& i : m_commonIncludes)
			source << "#line 1 " << curFile++ << '\n' << i.content << '\n';
		for(auto& i : m_localIncludes)
			source << "#line 1 " << curFile++ << '\n' << i.content << '\n';
		const auto result = source.str();
		const auto* pResult = result.c_str();

        glShaderSource(shader, 1, &pResult, nullptr);

        glCompileShader(shader);
        GLint isCompiled = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

		if(isCompiled)
		{
			m_localIncludes.clear();
			return *this;
		}
        // Read out the error message.
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string errorLog;
        errorLog.resize(length);
        glGetShaderInfoLog(shader, length, &length, &errorLog[0]);

        throw std::runtime_error("failed to compile shader:\n"
        // helper function to add debug names to the error log
		 + convertLog(errorLog, [this](GLint i) {
				if(i < GLint(m_commonIncludes.size()))
					return m_commonIncludes.at(i).debugName;
				return m_localIncludes.at(i - m_commonIncludes.size()).debugName;
			 }));
    }

    gl::Program ProgramBuilder::build_program() {
		gl::Program id(glCreateProgram());

        // attach shader
        for(auto& a : m_attachments)
			glAttachShader(id, a);

        // link shader
		glLinkProgram(id);

        // free shader attachments
		for(auto& a : m_attachments) {
			glDetachShader(id, a);
		}

        // verify linking
		GLint isLinked = GL_FALSE;
		glGetProgramiv(id, GL_LINK_STATUS, &isLinked);

        if(!isLinked) {
			GLint length = 0;
			glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);
			std::string errorLog;
			errorLog.resize(length);
			glGetProgramInfoLog(id, length, &length, &errorLog[0]);

			throw std::runtime_error("failed to link program " + errorLog);
        }

		return id;
    }

    void ProgramBuilder::add_include(IncludeFile file, bool isCommon) {
		if(isCommon) m_commonIncludes.emplace_back(std::move(file));
		else m_localIncludes.emplace_back(std::move(file));
    }
}
