#include "box_pipeline.hpp"
#include "core/opengl/program_builder.hpp"
#include "core/opengl/vertex_array_builder.hpp"
#include "core/opengl/gl_context.hpp"
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

	m_countProgramEx = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_define("SINGLE_MODEL")
		.add_define("SHADE_LEVEL")
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

	m_colorProgramEx = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_define("SINGLE_MODEL")
		.add_define("SHADE_LEVEL")
		.add_file("shader/box_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/box_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/box_fragment_color.glsl", false)
		.build_shader(gl::ShaderType::Fragment)
		.build_program();

	// vao only positions
	m_vao = gl::VertexArrayBuilder().add(0, 0, 3).build();
	m_vaoExt = gl::VertexArrayBuilder()
		.add(0, 0, 3)
		.add(1, 1, 1, false)
		.build();

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

	m_countPipeEx = m_colorPipe;
	m_countPipeEx.program = m_countProgramEx;
	m_countPipeEx.vertexArray = m_vaoExt;

	m_colorPipeEx = m_colorPipe;
	m_colorPipeEx.program = m_colorProgramEx;
	m_colorPipeEx.vertexArray = m_vaoExt;
}

void BoxPipeline::set_level_highlight(int levelIdx)
{
	m_levelHighlightIndex = levelIdx;
}

void BoxPipeline::draw(gl::Handle box, gl::Handle levels, int numBoxes, int numLevel, bool countingPass,
	const ei::Vec3& color) const
{
	draw(box, levels, ei::Mat3x4(ei::identity4x4()), numBoxes, numLevel, countingPass, color);
}

void BoxPipeline::draw(gl::Handle box, gl::Handle levels, ei::Mat3x4 transforms, int numBoxes, int numLevel,
	bool countingPass, const ei::Vec3& color) const
{
	if(countingPass)
	{
		gl::Context::set(m_countPipeEx);
	}
	else
	{
		gl::Context::set(m_colorPipeEx);
		glUniform3f(2, color.r, color.g, color.b);
		glUniform1i(10, numLevel);
		glUniform1i(11, m_levelHighlightIndex);
	}

	// transform matrix
	glUniformMatrix4x3fv(3, 1, GL_TRUE, reinterpret_cast<float*>(&transforms));

	// assumption about bbox layout
	static_assert(sizeof(ei::Box) == sizeof(ei::Vec3) * 2);
	mAssert(box);
	glBindVertexBuffer(0, box, 0, sizeof(ei::Vec3));

	mAssert(levels);
	glBindVertexBuffer(1, levels, 0, sizeof(int));

	glDrawArrays(GLenum(m_countPipe.topology), 0, numBoxes * 2);
}

void BoxPipeline::draw(gl::Handle box, const ArrayDevHandle_t<Device::OPENGL, ei::Mat3x4>& transforms, uint32_t numBoxes, bool countingPass, const ei::Vec3& color) const
{
	if (countingPass)
	{
		gl::Context::set(m_countPipe);
	}
	else
	{
		gl::Context::set(m_colorPipe);
		glUniform3f(2, color.r, color.g, color.b);
	}

	mAssert(transforms.id);
	mAssert(!transforms.offset);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, transforms.id);

	// assumption about bbox layout
	static_assert(sizeof(ei::Box) == sizeof(ei::Vec3) * 2);
	mAssert(box);
	glBindVertexBuffer(0, box, 0, sizeof(ei::Vec3));

	glDrawArrays(GLenum(m_countPipe.topology), 0, numBoxes * 2);
}
}
