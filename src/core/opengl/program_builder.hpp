#pragma once
#include "gl_object.hpp"
#include <string>
#include <vector>

namespace mufflon::gl {

enum class ShaderType {
    Vertex = 0x8B31,
    Fragment = 0x8B30,
    Geometry = 0x8DD9,
    TessControl = 0x8E88,
    TessEval = 0x8E87,
    Compute = 0x91B9
};

class ProgramBuilder {
public:
    // version: version number without #version prefix
	ProgramBuilder(std::string version = "460");

    // creates program from file
    // isCommon: indicates if this source will be used for all shaders (true) or only for the next shader (false)
	ProgramBuilder& add_file(const std::string& filename, bool isCommon = true);
    
    // creates program from string source
	// isCommon: indicates if this source will be used for all shaders (true) or only for the next shader (false)
	ProgramBuilder& add_source(std::string source, std::string debugName, bool isCommon = true);

    // adds a #define name
	// isCommon: indicates if this source will be used for all shaders (true) or only for the next shader (false)
    ProgramBuilder& add_define(std::string name, bool isCommon = true);

    // adds a #define name value
	// isCommon: indicates if this source will be used for all shaders (true) or only for the next shader (false)
    template<class T>
	ProgramBuilder& add_define(std::string name, const T& value, bool isCommon = true) {
		return add_define(name + " " + std::to_string(value), isCommon);
    }

    // creates a shader based on the supplied files/definitions.
    // the file list with isCommon = false will be cleared after this call
	ProgramBuilder& build_shader(ShaderType type);

	// returns the program handle
    gl::Program build_program();
private:
    struct IncludeFile {
		std::string debugName;
		std::string content;
    };

	void add_include(IncludeFile file, bool isCommon);

	std::vector<gl::Shader> m_attachments;
	std::vector<IncludeFile> m_commonIncludes;
	std::vector<IncludeFile> m_localIncludes;
	std::string m_version;
};
}
