#include "gl_forward.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.hpp"
#include "core/opengl/vertex_array_builder.hpp"
#include "core/export/interface.h"

namespace mufflon::renderer
{
	GlForward::GlForward() :
		GlRendererBase(true, false)
	{
	}

	void GlForward::post_reset()
	{
		init();

		GlRendererBase::post_reset();

		m_trianglePipe.framebuffer = m_framebuffer;
		m_quadPipe.framebuffer = m_framebuffer;
		m_spherePipe.framebuffer = m_framebuffer;

		glGenBuffers(1, &m_transformBuffer);
		glBindBuffer(GL_UNIFORM_BUFFER, m_transformBuffer);
		auto curTransforms = get_camera_transforms();
		glNamedBufferStorage(m_transformBuffer, sizeof(CameraTransforms), &curTransforms, 0);
	}

	void GlForward::init()
	{
		auto ltcGGX1 = reinterpret_cast<scene::textures::Texture*>(world_add_texture(
			"resources/ltc/ltc_ggx_1.dds",
			TextureSampling::SAMPLING_LINEAR,
			MipmapType::MIPMAP_NONE,
			nullptr, nullptr
		));
		auto ltcGGX2 = reinterpret_cast<scene::textures::Texture*>(world_add_texture(
			"resources/ltc/ltc_ggx_2.dds",
			TextureSampling::SAMPLING_LINEAR,
			MipmapType::MIPMAP_NONE,
			nullptr, nullptr
		));
		auto ltcBeckmann1 = reinterpret_cast<scene::textures::Texture*>(world_add_texture(
			"resources/ltc/ltc_beckmann_1.dds",
			TextureSampling::SAMPLING_LINEAR,
			MipmapType::MIPMAP_NONE,
			nullptr, nullptr
		));
		auto ltcBeckmann2 = reinterpret_cast<scene::textures::Texture*>(world_add_texture(
			"resources/ltc/ltc_beckmann_2.dds",
			TextureSampling::SAMPLING_LINEAR,
			MipmapType::MIPMAP_NONE,
			nullptr, nullptr
		));
		mAssert(ltcGGX1);
		mAssert(ltcGGX2);
		mAssert(ltcBeckmann1);
		mAssert(ltcBeckmann2);

		m_triangleProgram = gl::ProgramBuilder()
		                    .add_file("shader/camera_transforms.glsl")
		                    .add_file("shader/model_transforms.glsl")
		                    .add_file("shader/light_transforms.glsl")
		                    .add_file("shader/forward_vertex.glsl", false)
		                    .build_shader(gl::ShaderType::Vertex)
		                    .add_file("shader/material_id_binding.glsl", false)
		                    .add_file("shader/forward_tese.glsl", false)
		                    .build_shader(gl::ShaderType::TessEval)
		                    .add_file("shader/ltc.glsl", false)
		                    .add_file("shader/forward_shade.glsl", false)
		                    .add_file("shader/forward_fragment.glsl", false)
		                    .build_shader(gl::ShaderType::Fragment)
		                    .build_program();

		// add intermediate tesselation
		m_quadProgram = gl::ProgramBuilder()
		                .add_file("shader/camera_transforms.glsl")
		                .add_file("shader/model_transforms.glsl")
		                .add_file("shader/light_transforms.glsl")
		                .add_file("shader/forward_vertex.glsl", false)
		                .build_shader(gl::ShaderType::Vertex)
		                .add_file("shader/material_id_binding.glsl", false)
		                .add_file("shader/forward_quad_tese.glsl", false)
		                .build_shader(gl::ShaderType::TessEval)
		                .add_file("shader/ltc.glsl", false)
		                .add_file("shader/forward_shade.glsl", false)
		                .add_file("shader/forward_fragment.glsl", false)
		                .build_shader(gl::ShaderType::Fragment)
		                .build_program();

		m_sphereProgram = gl::ProgramBuilder()
		                  .add_file("shader/camera_transforms.glsl")
		                  .add_file("shader/model_transforms.glsl")
		                  .add_file("shader/light_transforms.glsl")
		                  .add_file("shader/sphere_vertex.glsl", false)
		                  .build_shader(gl::ShaderType::Vertex)
		                  .add_file("shader/sphere_geom.glsl", false)
		                  .build_shader(gl::ShaderType::Geometry)
		                  .add_file("shader/ltc.glsl", false)
		                  .add_file("shader/forward_shade.glsl", false)
		                  .add_file("shader/sphere_fragment.glsl", false)
		                  .build_shader(gl::ShaderType::Fragment)
		                  .build_program();

		// vertex layout
		m_triangleVao = gl::VertexArrayBuilder()
		                .add(0, 0, 3) // position
		                .add(1, 1, 3) // normals
		                .add(2, 2, 2) // texcoords
		                .build();

		m_spheresVao = gl::VertexArrayBuilder()
		               .add(0, 0, 3, true, sizeof(float), 0) // position
		               .add(0, 1, 1, true, sizeof(float), 3 * sizeof(float)) // radius
		               .add(1, 2, 1, false, sizeof(u16)) // material indices
		               .build();

		// pipelines
		m_trianglePipe.program = m_triangleProgram;
		m_trianglePipe.vertexArray = m_triangleVao;
		m_trianglePipe.depthStencil.depthTest = true;
		m_trianglePipe.topology = gl::PrimitiveTopology::Patches;
		m_trianglePipe.rasterizer.cullMode = gl::CullMode::None;
		m_trianglePipe.rasterizer.frontFaceWinding = gl::Winding::CW;

		m_quadPipe = m_trianglePipe;
		m_quadPipe.patch.vertices = 4;
		m_quadPipe.program = m_quadProgram;

		m_spherePipe = m_trianglePipe;
		m_spherePipe.program = m_sphereProgram;
		m_spherePipe.vertexArray = m_spheresVao;
		m_spherePipe.topology = gl::PrimitiveTopology::Points;

		// set uniforms
		// ltc data
		const auto ltcTexHdl = ltcGGX1->acquire_const<DEVICE>();
		glProgramUniformHandleui64ARB(m_triangleProgram, 24, ltcTexHdl);
		glProgramUniformHandleui64ARB(m_sphereProgram, 24, ltcTexHdl);
		glProgramUniformHandleui64ARB(m_quadProgram, 24, ltcTexHdl);

		const auto ltcTexHdl2 = ltcGGX2->acquire_const<DEVICE>();
		glProgramUniformHandleui64ARB(m_triangleProgram, 25, ltcTexHdl2);
		glProgramUniformHandleui64ARB(m_sphereProgram, 25, ltcTexHdl2);
		glProgramUniformHandleui64ARB(m_quadProgram, 25, ltcTexHdl2);

		const auto ltcBeckmannHdl1 = ltcBeckmann1->acquire_const<DEVICE>();
		glProgramUniformHandleui64ARB(m_triangleProgram, 26, ltcBeckmannHdl1);
		glProgramUniformHandleui64ARB(m_sphereProgram, 26, ltcBeckmannHdl1);
		glProgramUniformHandleui64ARB(m_quadProgram, 26, ltcBeckmannHdl1);

		const auto ltcBeckmannHdl2 = ltcBeckmann2->acquire_const<DEVICE>();
		glProgramUniformHandleui64ARB(m_triangleProgram, 27, ltcBeckmannHdl2);
		glProgramUniformHandleui64ARB(m_sphereProgram, 27, ltcBeckmannHdl2);
		glProgramUniformHandleui64ARB(m_quadProgram, 27, ltcBeckmannHdl2);
	}

	void GlForward::iterate()
	{
		begin_frame({0.0f, 0.0f, 0.0f, 1.0f});

		// camera matrices
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_transformBuffer);

		draw_triangles(m_trianglePipe, Attribute::All);
		draw_quads(m_quadPipe, Attribute::All);
		draw_spheres(m_spherePipe, Attribute::All);

		end_frame();
	}
} // namespace mufflon::renderer
