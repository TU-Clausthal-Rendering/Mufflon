#include "gl_wireframe.h"
#include "core/scene/scene.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/gl_context.h"

namespace mufflon::renderer {

GlWireframe::GlWireframe() :
    GlRendererBase(true, false)
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
        .add_file("shader/wireframe_quad_geom.glsl", false)
        .build_shader(gl::ShaderType::Geometry)
		.add_file("shader/wireframe_fragment.glsl", false)
        .build_shader(gl::ShaderType::Fragment)
		.build_program();

    m_quadDepthProgram = gl::ProgramBuilder()
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
		.add_file("shader/sphere_vertex.glsl", false)
        .build_shader(gl::ShaderType::Vertex)
		.add_file("shader/wireframe_sgeom.glsl", false)
        .build_shader(gl::ShaderType::Geometry)
		.add_file("shader/wireframe_fragment.glsl", false)
        .build_shader(gl::ShaderType::Fragment)
		.build_program();

    m_sphereDepthProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/sphere_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/sphere_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/sphere_fragment.glsl", false)
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
	m_trianglePipe.depthStencil.depthCmpFunc = gl::CmpFunc::LessEqual;

	m_quadPipe = m_trianglePipe;
	m_quadPipe.patch.vertices = 4;
	m_quadPipe.program = m_quadProgram;

	m_spherePipe = m_trianglePipe;
	m_spherePipe.program = m_sphereProgram;
	m_spherePipe.vertexArray = m_spheresVao;

    // depth pre pass pipeline
	m_triangleDepthPipe.program = m_triangleProgram;
	m_triangleDepthPipe.vertexArray = m_triangleVao;
	m_triangleDepthPipe.rasterizer.cullMode = gl::CullMode::None;
	m_triangleDepthPipe.depthStencil.depthTest = true;
	m_triangleDepthPipe.rasterizer.colorWrite = false;
    // add offset to prevent z fighting that occurs because primitive types (lines and triangles) are mixed
	m_triangleDepthPipe.depthStencil.polygonOffsetFactor = 1.0f;
	m_triangleDepthPipe.depthStencil.polygonOffsetUnits = 1.0f;

	m_quadDepthPipe = m_triangleDepthPipe;
	m_quadDepthPipe.patch.vertices = 4;
	m_quadDepthPipe.program = m_quadDepthProgram;

	m_sphereDepthPipe = m_triangleDepthPipe;
	m_sphereDepthPipe.program = m_sphereDepthProgram;
	m_sphereDepthPipe.vertexArray = m_spheresVao;
}

void GlWireframe::on_reset() {
	GlRendererBase::on_reset();
    // set framebuffer
	m_trianglePipe.framebuffer = m_framebuffer;
	m_quadPipe.framebuffer = m_framebuffer;
	m_spherePipe.framebuffer = m_framebuffer;
	m_triangleDepthPipe.framebuffer = m_framebuffer;
	m_quadDepthPipe.framebuffer = m_framebuffer;
	m_sphereDepthPipe.framebuffer = m_framebuffer;

    // apply parameters
	m_trianglePipe.rasterizer.lineWidth = m_params.lineWidth;
	m_trianglePipe.depthStencil.depthTest = m_params.enableDepth;
	m_quadPipe.rasterizer.lineWidth = m_params.lineWidth;
	m_quadPipe.depthStencil.depthTest = m_params.enableDepth;
	m_spherePipe.rasterizer.lineWidth = m_params.lineWidth;
	m_spherePipe.depthStencil.depthTest = m_params.enableDepth;

	glGenBuffers(1, &m_transformBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_transformBuffer);
	auto curTransforms = get_camera_transforms();
	glNamedBufferStorage(m_transformBuffer, sizeof(CameraTransforms), &curTransforms, 0);

    // set uniform parameters
	glProgramUniform2f(m_sphereProgram, 2, float(m_outputBuffer.get_width()), float(m_outputBuffer.get_height()));
}

void GlWireframe::iterate() {
	begin_frame({ 0.0f, 0.0f, 0.0f, 1.0f });
	
    // camera matrices
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_transformBuffer);

    if(m_params.enableDepth) {
		draw_triangles(m_triangleDepthPipe, Attribute::Position);
		draw_quads(m_quadDepthPipe, Attribute::Position);
		draw_spheres(m_sphereDepthPipe, Attribute::Position);
    }

	draw_triangles(m_trianglePipe, Attribute::Position);
	draw_quads(m_quadPipe, Attribute::Position);
	draw_spheres(m_spherePipe, Attribute::Position);

	end_frame();
}
}
