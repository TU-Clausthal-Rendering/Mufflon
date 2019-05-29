#include "gl_wireframe.h"
#include "core/scene/scene.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"

namespace mufflon::renderer {

GlWireframe::GlWireframe() {
    // shader
	m_triangleProgram = gl::ProgramBuilder()
        .add_file(gl::ShaderType::Vertex, "shader/wireframe_vertex.glsl")
        .add_file(gl::ShaderType::Fragment, "shader/wireframe_fragment.glsl")
        .build();

	m_quadProgram = gl::ProgramBuilder()
		.add_file(gl::ShaderType::Vertex, "shader/wireframe_vertex.glsl")
		.add_file(gl::ShaderType::TessEval, "shader/wireframe_tese.glsl")
		.add_file(gl::ShaderType::Fragment, "shader/wireframe_fragment.glsl")
		.build();

    // vertex layout
	m_vao = gl::VertexArrayBuilder().add(
		0, 0, 3 // Positions
	).build();

    // wireframe pipeline
	m_trianglePipe.program = m_triangleProgram;
	m_trianglePipe.vertexArray = m_vao;
	m_trianglePipe.rasterizer.cullMode = gl::CullMode::None;
	m_trianglePipe.rasterizer.fillMode = gl::FillMode::Wireframe;

	m_quadPipe = m_trianglePipe;
	m_quadPipe.patch.vertices = 4;
	m_quadPipe.program = m_quadProgram;
}

void GlWireframe::on_reset() {
	GlRendererBase::on_reset();

	m_trianglePipe.framebuffer = m_framebuffer;
	m_quadPipe.framebuffer = m_framebuffer;

	auto* cam = m_currentScene->get_camera();
	float fov = 1.5f;
    if(auto pcam = dynamic_cast<const cameras::Pinhole*>(cam)) {
		fov = pcam->get_vertical_fov();
	}
    m_viewProjMatrix = 
		ei::perspectiveGL(fov, 
		    float(m_outputBuffer.get_width()) / m_outputBuffer.get_height(), 
		    cam->get_near(), cam->get_far()) * 
		ei::camera(
			cam->get_position(0),
            cam->get_position(0) + cam->get_view_dir(0),
            cam->get_up_dir(0)
		);
}

void GlWireframe::iterate() {
	begin_frame({ 0.0f, 0.0f, 0.0f, 1.0f });
	
    // camera matrix
	glProgramUniformMatrix4fv(m_triangleProgram, 0, 1, GL_TRUE, reinterpret_cast<const float*>(&m_viewProjMatrix));
	glProgramUniformMatrix4fv(m_quadProgram, 0, 1, GL_TRUE, reinterpret_cast<const float*>(&m_viewProjMatrix));

	draw_triangles(m_trianglePipe, Attribute::Position);
	draw_quads(m_quadPipe, Attribute::Position);

	end_frame();
}
}
