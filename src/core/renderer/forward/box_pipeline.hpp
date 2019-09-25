#pragma once
#include "core/opengl/gl_object.h"
#include "core/opengl/gl_pipeline.h"
#include "core/memory/residency.hpp"
#include "ei/3dtypes.hpp"

namespace mufflon::renderer {
	
// helper class to draw bounding boxes with transparency
class BoxPipeline {
public:
	BoxPipeline();
	void init(gl::Framebuffer& framebuffer);
	void draw(const ArrayDevHandle_t<Device::OPENGL, ei::Box>& box, uint32_t numBoxes) const;
private:
	gl::Program m_program;
	gl::Pipeline m_pipe;
	gl::VertexArray m_vao;
};
}
