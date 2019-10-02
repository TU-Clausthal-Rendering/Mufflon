#pragma once
#include "gl_object.hpp"

namespace mufflon::gl {

class VertexArrayBuilder {
public:
	VertexArrayBuilder();
	VertexArrayBuilder& add(
		uint32_t cpuBinding, // cpu buffer binding
		uint32_t gpuLocation, // glsl layout location
		int numComponents,
        bool isFloat = true, // otherwise integer
        uint32_t componentByteSize = 4,
        uint32_t byteOffset = 0, // byte offset to the first element
        uint32_t divisor = 0
	);
	gl::VertexArray build();
private:
	gl::VertexArray m_id;
};
}