#include "box_pipeline.hpp"
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"
#include <glad/glad.h>

namespace mufflon::renderer {

BoxPipeline::BoxPipeline() {

}

void BoxPipeline::init(gl::Framebuffer& framebuffer) {
	m_program = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/box_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/box_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/box_fragment.glsl", false)
		.build_shader(gl::ShaderType::Fragment)	
		.build_program();

	// vao only positions
	m_vao = gl::VertexArrayBuilder().add(0, 0, 3).build();

	m_pipe.framebuffer = framebuffer;
	m_pipe.program = m_program;
	m_pipe.vertexArray = m_vao;
	m_pipe.depthStencil.depthTest = true;
	m_pipe.depthStencil.depthWrite = false;
	m_pipe.topology = gl::PrimitiveTopology::Lines; // bbox max and bbox min points
	m_pipe.rasterizer.cullMode = gl::CullMode::None;

	m_pipe.rasterizer.fillMode = gl::FillMode::Wireframe;
}

void BoxPipeline::draw(const ArrayDevHandle_t<Device::OPENGL, ei::Box>& box, uint32_t numBoxes) const
{
	gl::Context::set(m_pipe);

	// assumption about bbox layout
	static_assert(sizeof(ei::Box) == sizeof(ei::Vec3) * 2);
	mAssert(box.id);
	glBindVertexBuffer(0, box.id, box.offset, sizeof(ei::Vec3));
	
	glDrawArrays(GLenum(m_pipe.topology), 0, numBoxes * 2);
}
}
