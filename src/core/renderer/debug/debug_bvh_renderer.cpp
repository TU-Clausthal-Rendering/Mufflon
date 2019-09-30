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
	const auto attribs = Attribute::Normal | Attribute::Position | Attribute::Texcoord | Attribute::Material;
	draw_triangles(m_trianglePipe, attribs);
	draw_quads(m_quadPipe, attribs);
	draw_spheres(m_spherePipe, attribs);

	const auto& sceneDesc = this->get_scene_descriptor();

	static const ei::Vec3 TopColor = ei::Vec3(1.0f, 0.5f, 1.0f);
	static const ei::Vec3 BotColor = ei::Vec3(0.0f, 0.5f, 1.0f);
	static const ei::Vec3 BoxColor = ei::Vec3(0.7f, 0.0f, 0.07f);

	// count transparent fragments
	m_dynFragmentBuffer.bindCountBuffer();
	if (m_showBoxes)
		m_boxPipe.draw(m_bboxes, sceneDesc.instanceToWorld, sceneDesc.numInstances, true, BoxColor);
	if (m_showTopLevel)
		m_boxPipe.draw(m_topLevelBoxes, m_topLevelNumBoxes, true, TopColor);
	if (m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, sceneDesc.instanceToWorld, m_botIdx, m_botLevelNumBoxes, true, BotColor);
	
	
	m_dynFragmentBuffer.prepareFragmentBuffer();
	//// draw transparent fragments with color
	if (m_showBoxes)
		m_boxPipe.draw(m_bboxes, sceneDesc.instanceToWorld, sceneDesc.numInstances, false, BoxColor);
	if (m_showTopLevel)
		m_boxPipe.draw(m_topLevelBoxes, m_topLevelNumBoxes, false, TopColor);
	if (m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, sceneDesc.instanceToWorld, m_botIdx, m_botLevelNumBoxes, false, BotColor);
	
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
	m_showBoxes = this->m_params.get_param_bool(PDebugBoxes::name);
	m_showTopLevel = this->m_params.get_param_bool(PDebugTopLevel::name);
	m_botIdx = this->m_params.get_param_int(PDebugBotLevel::name);
	m_showBotLevel = m_botIdx >= 0 && m_botIdx < sceneDesc.numInstances;

	if(m_showTopLevel)
		upload_box_array(sceneDesc.cpuDescriptor->accelStruct, m_topLevelBoxes, m_topLevelNumBoxes, sceneDesc.aabb);

	if (m_showBotLevel)
		upload_box_array(sceneDesc.cpuDescriptor->lods[m_botIdx].accelStruct, m_botLevelBoxes, m_botLevelNumBoxes, sceneDesc.cpuDescriptor->aabbs[m_botIdx]);

	if(m_showBoxes)
	{
		std::vector<ei::Box> bbox;
		bbox.reserve(sceneDesc.numInstances);
		for(int i = 0; i < sceneDesc.numInstances; ++i)
		{
			auto lodIdx = sceneDesc.lodIndices[i];
			const auto& lod = sceneDesc.lods[lodIdx];
			bbox.push_back(sceneDesc.cpuDescriptor->aabbs[lodIdx]);
		}

		glGenBuffers(1, &m_bboxes);
		gl::bindBuffer(gl::BufferType::ShaderStorage, m_bboxes);
		gl::bufferStorage(m_bboxes, bbox.size() * sizeof(bbox[0]), bbox.data(), gl::StorageFlags::None);
	}
}

void mufflon::renderer::DebugBvhRenderer::init()
{
	// shader must be reloaded for changed color coding
	//if (m_isInit) return;
	//m_isInit = true;
	const auto& sceneDesc = this->get_scene_descriptor();
	auto colorCoding = this->m_params.get_param_bool(PDebugColorInstance::name);

	m_triangleProgram = gl::ProgramBuilder()
		.add_define("COLOR_INSTANCE", colorCoding, true)
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
		.add_define("COLOR_INSTANCE", colorCoding, true)
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
		.add_define("COLOR_INSTANCE", colorCoding, true)
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

void mufflon::renderer::DebugBvhRenderer::upload_box_array(const scene::AccelDescriptor& accel, gl::Buffer& dstBuffer,
	int& boxCount, const ei::Box& root) {
	// create array with bounding boxes
	auto blas = reinterpret_cast<const scene::accel_struct::LBVH<Device::CPU>*>(accel.accelParameters);
	mAssert(blas);

	std::vector<ei::Box> bboxes;
	//std::vector<scene::accel_struct::BvhNode> tst;

	boxCount = blas->numInternalNodes * 2;
	if (boxCount == 0)
	{
		// special case => show top level box
		bboxes.push_back(root);
		boxCount = 1;
	}
	else
	{
		bboxes.resize(boxCount);
		for (int i = 0; i < boxCount; ++i) {
			bboxes[i] = blas->bvh[i].bb;
		}
	}

	glGenBuffers(1, &dstBuffer);
	gl::bindBuffer(gl::BufferType::ShaderStorage, dstBuffer);
	gl::bufferStorage(dstBuffer, bboxes.size() * sizeof(bboxes[0]), bboxes.data(), gl::StorageFlags::None);
}
