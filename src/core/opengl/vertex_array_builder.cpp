#include "util/assert.hpp"
#include "vertex_array_builder.hpp"
#include <glad/glad.h>

namespace mufflon::gl {

    VertexArrayBuilder::VertexArrayBuilder() {
		glGenVertexArrays(1, &m_id);
		glBindVertexArray(m_id);
    }

	static GLenum get_float_type(uint32_t byteSize) {
        switch (byteSize) {
		case 2: return GL_HALF_FLOAT;
		case 4: return GL_FLOAT;
		//case 8: return GL_DOUBLE; // only works with attribL
		default: mAssert(false);
        }
		return 0;
    }

    static GLenum get_int_type(uint32_t byteSize) {
        switch (byteSize) {
		case 1: return GL_BYTE;
		case 2: return GL_SHORT;
		case 4: return GL_INT;
		default: mAssert(false);
        }
		return 0;
    }

    VertexArrayBuilder& VertexArrayBuilder::add(uint32_t cpuBinding, uint32_t gpuLocation, int numComponents,
        bool isFloat, uint32_t componentByteSize, uint32_t byteOffset, uint32_t divisor) {
        
		glEnableVertexArrayAttrib(m_id, gpuLocation);
		glVertexArrayAttribBinding(m_id, gpuLocation, cpuBinding);
		glVertexArrayBindingDivisor(m_id, cpuBinding, divisor);
		if(isFloat)
			if(componentByteSize == 8)
				glVertexArrayAttribLFormat(m_id, gpuLocation, numComponents, GL_DOUBLE, byteOffset);
			else
			    glVertexArrayAttribFormat(m_id, gpuLocation, numComponents, get_float_type(componentByteSize), GL_TRUE, byteOffset);
		else
			glVertexArrayAttribIFormat(m_id, gpuLocation, numComponents, get_int_type(componentByteSize), byteOffset);

		return *this;
    }

    gl::VertexArray VertexArrayBuilder::build() {
		return std::move(m_id);
    }
}
