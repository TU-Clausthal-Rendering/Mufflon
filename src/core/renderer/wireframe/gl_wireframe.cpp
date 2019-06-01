#include "gl_wireframe.h"
#include "core/scene/scene.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"

namespace mufflon::renderer {

GlWireframe::GlWireframe() :
    GlRendererBase(false, false)
{
    // shader
	m_triangleProgram = gl::ProgramBuilder()
        .add_file("shader/camera_transforms.glsl")
        .add_file("shader/wireframe_vertex.glsl", false)
        .build_shader(gl::ShaderType::Vertex)
        .add_file("shader/wireframe_fragment.glsl", false)
        .build_shader(gl::ShaderType::Fragment)
        .build_program();

	m_quadProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/wireframe_vertex.glsl", false)
        .build_shader(gl::ShaderType::Vertex)
		.add_file("shader/wireframe_tese.glsl", false)
        .build_shader(gl::ShaderType::TessEval)
		.add_file("shader/wireframe_fragment.glsl", false)
        .build_shader(gl::ShaderType::Fragment)
		.build_program();

	m_sphereProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/wireframe_svertex.glsl", false)
        .build_shader(gl::ShaderType::Vertex)
		.add_file("shader/wireframe_sgeom.glsl", false)
        .build_shader(gl::ShaderType::Geometry)
		.add_file("shader/wireframe_fragment.glsl", false)
        .build_shader(gl::ShaderType::Fragment)
		.build_program();

    // vertex layout
	m_triangleVao = gl::VertexArrayBuilder()
        .add(0, 0, 3) // position
        .build();

	m_spheresVao = gl::VertexArrayBuilder()
        .add(0, 0, 3, true, 4, 0) // position
        .add(0, 1, 1, true, 4, 3 * sizeof(float)) // radius
        .build();

    // wireframe pipeline
	m_trianglePipe.program = m_triangleProgram;
	m_trianglePipe.vertexArray = m_triangleVao;
	m_trianglePipe.rasterizer.cullMode = gl::CullMode::None;
	m_trianglePipe.rasterizer.fillMode = gl::FillMode::Wireframe;

	m_quadPipe = m_trianglePipe;
	m_quadPipe.patch.vertices = 4;
	m_quadPipe.program = m_quadProgram;

	m_spherePipe = m_trianglePipe;
	m_spherePipe.program = m_sphereProgram;
	m_spherePipe.vertexArray = m_spheresVao;
}

void GlWireframe::on_reset() {
	GlRendererBase::on_reset();

	m_trianglePipe.framebuffer = m_framebuffer;
	m_quadPipe.framebuffer = m_framebuffer;
	m_spherePipe.framebuffer = m_framebuffer;

	glGenBuffers(1, &m_transformBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_transformBuffer);
	auto curTransforms = get_camera_transforms();
	glNamedBufferStorage(m_transformBuffer, sizeof(CameraTransforms), &curTransforms, 0);
}

void GlWireframe::iterate() {
	begin_frame({ 0.0f, 0.0f, 0.0f, 1.0f });
	
    // camera matrices
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_transformBuffer);

	draw_triangles(m_trianglePipe, Attribute::Position);
	draw_quads(m_quadPipe, Attribute::Position);
	draw_spheres(m_spherePipe);

	end_frame();
}
}
