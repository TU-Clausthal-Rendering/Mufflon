#pragma once
#include "gl_wrapper.hpp"
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
	~ProgramBuilder();
    // creates program from file
	ProgramBuilder& add_file(ShaderType type, const std::string& filename);
    // creates program from string source
	ProgramBuilder& add_source(ShaderType type, const std::string& source, const std::string& debugName = "");
	// returns the program handle
    gl::Handle build();
private:
	void unload(gl::Handle program);
private:
	std::vector<gl::Handle> m_attachments;
};
}
