#include "gl_wireframe.h"
#include "core/scene/scene.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"

namespace mufflon::renderer {

GlWireframe::GlWireframe() {
    // shader
	m_program = gl::ProgramBuilder().add_source(gl::ShaderType::Vertex, R"(
#version 460
layout(location = 0) in vec3 in_position;        

layout(location = 0) uniform mat4 u_viewProj;

void main(){
    gl_Position = u_viewProj * vec4(in_position, 1.0);
}

    )").add_source(gl::ShaderType::Fragment, R"(
#version 460
layout(location = 0) out vec4 out_fragColor;   
void main(){
    out_fragColor = vec4(1.0);
}

    )").build();

    // vertex layout
	m_vao = gl::VertexArrayBuilder().add(
		0, 0, 3 // Positions
	).build();

    // wireframe pipeline
	m_pipe.program = m_program;
	m_pipe.vertexArray = m_vao;
	m_pipe.rasterizer.cullMode = gl::CullMode::None;
	m_pipe.rasterizer.fillMode = gl::FillMode::Wireframe;
}

void GlWireframe::on_reset() {
	GlRendererBase::on_reset();

	m_pipe.framebuffer = m_framebuffer;
    
	auto* cam = m_currentScene->get_camera();
	m_viewProjMatrix = 
		ei::perspectiveGL(1.5f, 
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
	glProgramUniformMatrix4fv(m_program, 0, 1, GL_TRUE, reinterpret_cast<const float*>(&m_viewProjMatrix));

	draw_triangles(m_pipe, Attribute::Position);


	end_frame();
}
}
