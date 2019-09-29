#include "debug_bvh_renderer.hpp"
#include "glad/glad.h"
#include "core/opengl/vertex_array_builder.h"
#include "core/opengl/program_builder.h"

mufflon::renderer::DebugBvhRenderer::DebugBvhRenderer()
	: GlRendererBase(true, false)
{

}

void mufflon::renderer::DebugBvhRenderer::iterate()
{
	begin_frame({ 0.0f, 0.0f, 0.0f, 1.0f });

	// camera matrices
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_transformBuffer);

	// draw background geometry
	draw_triangles(m_trianglePipe, Attribute::All);
	draw_quads(m_quadPipe, Attribute::All);
	draw_spheres(m_spherePipe, Attribute::All);

	const auto& sceneDesc = this->get_scene_descriptor();

	// count transparent fragments
	m_dynFragmentBuffer.bindCountBuffer();
	if (m_showTopLevel)
		m_boxPipe.draw(sceneDesc.aabbs.id, sceneDesc.instanceToWorld, sceneDesc.numInstances, true);
	if (m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, sceneDesc.instanceToWorld, m_botIdx, m_botLevelNumBoxes, true);


	m_dynFragmentBuffer.prepareFragmentBuffer();
	//// draw transparent fragments with color
	if (m_showTopLevel)
		m_boxPipe.draw(sceneDesc.aabbs.id, sceneDesc.instanceToWorld, sceneDesc.numInstances, false);
	if (m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, sceneDesc.instanceToWorld, m_botIdx, m_botLevelNumBoxes, false);

	m_dynFragmentBuffer.blendFragmentBuffer();

	end_frame();
}

void mufflon::renderer::DebugBvhRenderer::post_reset()
{
	init();

	GlRendererBase::post_reset();

	m_trianglePipe.framebuffer = m_framebuffer;
	m_quadPipe.framebuffer = m_framebuffer;
	m_spherePipe.framebuffer = m_framebuffer;

	m_boxPipe.init(m_framebuffer);
	m_dynFragmentBuffer.init(m_framebuffer, m_outputBuffer.get_width(), m_outputBuffer.get_height());

	const auto& sceneDesc = this->get_scene_descriptor();
	m_showTopLevel = this->m_params.get_param_bool(PDebugTopLevel::name);
	m_botIdx = this->m_params.get_param_int(PDebugBotLevel::name);
	m_showBotLevel = m_botIdx >= 0 && m_botIdx < sceneDesc.numInstances;

	if (m_showBotLevel) {
		// create array with bounding boxes
		auto blas = reinterpret_cast<const scene::accel_struct::LBVH<Device::OPENGL>*>(sceneDesc.lods[m_botIdx].accelStruct.accelParameters);
		mAssert(blas);

		std::vector<ei::Box> bboxes;
		m_botLevelNumBoxes = blas->numInternalNodes;
		bboxes.resize(blas->numInternalNodes);
		for (int i = 0; i < blas->numInternalNodes; ++i) {
			bboxes[i] = blas->bvh[i].bb;
		}

		glGenBuffers(1, &m_botLevelBoxes);
		gl::bindBuffer(gl::BufferType::ShaderStorage, m_botLevelBoxes);
		gl::bufferStorage(m_botLevelBoxes, bboxes.size() * sizeof(bboxes[0]), bboxes.data(), gl::StorageFlags::None);
	}
}

void mufflon::renderer::DebugBvhRenderer::init()
{
	if (m_isInit) return;
	m_isInit = true;

	m_triangleProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/model_transforms.glsl")
		.add_file("shader/forward_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/material_id_binding.glsl", false)
		.add_file("shader/forward_tese.glsl", false)
		.build_shader(gl::ShaderType::TessEval)
		.add_file("shader/ndotc_shade.glsl", false)
		.add_file("shader/ndotc_fragment.glsl", false)
		.build_shader(gl::ShaderType::Fragment)
		.build_program();

	// add intermediate tesselation
	m_quadProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/model_transforms.glsl")
		.add_file("shader/forward_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/material_id_binding.glsl", false)
		.add_file("shader/forward_quad_tese.glsl", false)
		.build_shader(gl::ShaderType::TessEval)
		.add_file("shader/ndotc_shade.glsl", false)
		.add_file("shader/ndotc_fragment.glsl", false)
		.build_shader(gl::ShaderType::Fragment)
		.build_program();

	m_sphereProgram = gl::ProgramBuilder()
		.add_file("shader/camera_transforms.glsl")
		.add_file("shader/model_transforms.glsl")
		.add_file("shader/sphere_vertex.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/sphere_geom.glsl", false)
		.build_shader(gl::ShaderType::Geometry)
		.add_file("shader/ndotc_shade.glsl", false)
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

	// camera transforms buffer
	glGenBuffers(1, &m_transformBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, m_transformBuffer);
	auto curTransforms = get_camera_transforms();
	glNamedBufferStorage(m_transformBuffer, sizeof(CameraTransforms), &curTransforms, 0);
}
