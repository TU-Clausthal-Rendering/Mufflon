#include "box_pipeline.hpp"
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"
#include <glad/glad.h>

namespace mufflon::renderer {

BoxPipeline::BoxPipeline() {

}

void BoxPipeline::init(gl::Framebuffer& framebuffer) {
	m_countProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/model_transforms.glsl")
		.add_file("shader/box_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/box_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/box_fragment_count.glsl", false)
		.build_shader(gl::ShaderType::Fragment)	
		.build_program();

	m_colorProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/model_transforms.glsl")
		.add_file("shader/box_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/box_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/box_fragment_color.glsl", false)
		.build_shader(gl::ShaderType::Fragment)
		.build_program();

	// vao only positions
	m_vao = gl::VertexArrayBuilder().add(0, 0, 3).build();

	m_countPipe.framebuffer = framebuffer;
	m_countPipe.program = m_countProgram;
	m_countPipe.vertexArray = m_vao;
	m_countPipe.depthStencil.depthTest = true;
	m_countPipe.depthStencil.depthWrite = false;
	m_countPipe.rasterizer.colorWrite = false;
	m_countPipe.topology = gl::PrimitiveTopology::Lines; // bbox max and bbox min points
	m_countPipe.rasterizer.cullMode = gl::CullMode::None;
	m_countPipe.depthStencil.polygonOffsetUnits = -1.0f;
	m_countPipe.depthStencil.polygonOffsetFactor = -1.0f;

	m_colorPipe = m_countPipe;
	m_colorPipe.program = m_colorProgram;
}

void BoxPipeline::draw(const ArrayDevHandle_t<Device::OPENGL, ei::Box>& box, const ArrayDevHandle_t<Device::OPENGL, ei::Mat3x4>& transforms, uint32_t numBoxes, bool countingPass) const
{
	if(countingPass)
	{
		gl::Context::set(m_countPipe);
	}
	else
	{
		gl::Context::set(m_colorPipe);
	}

	// bind transforms
	mAssert(transforms.id);
	mAssert(!transforms.offset);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, transforms.id);

	// assumption about bbox layout
	static_assert(sizeof(ei::Box) == sizeof(ei::Vec3) * 2);
	mAssert(box.id);
	glBindVertexBuffer(0, box.id, box.offset, sizeof(ei::Vec3));
	
	glDrawArrays(GLenum(m_countPipe.topology), 0, numBoxes * 2);
}
}
