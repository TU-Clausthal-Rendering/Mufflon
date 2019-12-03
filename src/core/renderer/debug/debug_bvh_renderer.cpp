#include "debug_bvh_renderer.hpp"
#include "glad/glad.h"
#include "core/opengl/vertex_array_builder.hpp"
#include "core/opengl/program_builder.hpp"
#include <stack>

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
	const auto instanceToWorld = m_sceneDesc.cpuDescriptor->compute_instance_to_world_transformation(m_botIdx);
	if (m_showBoxes)
		m_boxPipe.draw(m_bboxes, sceneDesc.instanceToWorld, sceneDesc.numInstances, true, BoxColor);
	if (m_showTopLevel)
		m_boxPipe.draw(m_topLevelBoxes, m_topLevelLevels, m_topLevelNumBoxes, m_topLevelMaxLevel, true, TopColor);
	// TODO: let OpenGL fetch the instance transformation from the (OpenGL-side already present) descriptor?
	if(m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, m_botLevelLevels, instanceToWorld,
					   m_botLevelNumBoxes, m_botLevelMaxLevel, true, BotColor);
	
	
	m_dynFragmentBuffer.prepareFragmentBuffer();
	//// draw transparent fragments with color
	if (m_showBoxes)
		m_boxPipe.draw(m_bboxes, sceneDesc.instanceToWorld, sceneDesc.numInstances, false, BoxColor);
	if (m_showTopLevel)
		m_boxPipe.draw(m_topLevelBoxes, m_topLevelLevels, m_topLevelNumBoxes, m_topLevelMaxLevel, false, TopColor);
	if (m_showBotLevel)
		m_boxPipe.draw(m_botLevelBoxes, m_botLevelLevels, instanceToWorld,
			m_botLevelNumBoxes, m_botLevelMaxLevel, false, BotColor);
	
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
	auto lastShowTopLevel = m_showTopLevel;
	m_showTopLevel = this->m_params.get_param_bool(PDebugTopLevel::name);
	auto lastBotIdx = m_botIdx;
	m_botIdx = this->m_params.get_param_int(PDebugBotLevel::name);
	m_showBotLevel = m_botIdx >= 0 && m_botIdx < sceneDesc.numInstances;
	m_levelHighlightIdx = this->m_params.get_param_int(PDebugLevelHighlight::name);
	m_boxPipe.set_level_highlight(m_levelHighlightIdx);
	m_minLevel = this->m_params.get_param_int(PDebugMinLevel::name);
	m_maxLevel = this->m_params.get_param_int(PDebugMaxLevel::name);


	if (m_showTopLevel) {
		upload_box_array(sceneDesc.cpuDescriptor->accelStruct, m_topLevelBoxes, m_topLevelLevels, m_topLevelNumBoxes, m_topLevelMaxLevel);
		if (!m_topLevelNumBoxes) m_showTopLevel = false;
		if(lastShowTopLevel != m_showTopLevel)
			logInfo("top level count: ", m_topLevelMaxLevel);
	}

	if (m_showBotLevel) {
		upload_box_array(sceneDesc.cpuDescriptor->lods[m_botIdx].accelStruct, m_botLevelBoxes, m_botLevelLevels, m_botLevelNumBoxes, m_botLevelMaxLevel);
		if (!m_botLevelNumBoxes) m_showBotLevel = false;
		if(lastBotIdx != m_botIdx)
			logInfo("bottom level count: ", m_botLevelMaxLevel);
	}

	if(m_showBoxes)
	{
		std::vector<ei::Box> bbox;
		bbox.reserve(sceneDesc.numInstances);
		for(int i = 0; i < sceneDesc.numInstances; ++i)
		{
			auto lodIdx = sceneDesc.lodIndices[i];
			if(!sceneDesc.is_instance_present(lodIdx))
				continue;
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

void mufflon::renderer::DebugBvhRenderer::upload_box_array(const scene::AccelDescriptor& accel, gl::Buffer& dstBoxBuffer, gl::Buffer& dstLevelBuffer,
	int& boxCount, int& maxLevel) {
	// create array with bounding boxes
	auto blas = reinterpret_cast<const scene::accel_struct::LBVH<Device::CPU>*>(accel.accelParameters);
	mAssert(blas);

	std::vector<ei::Box> bboxes;
	std::vector<int> level;
	//std::vector<scene::accel_struct::BvhNode> tst;


	boxCount = blas->numInternalNodes * 2;
	if (!boxCount) return;

	
	bboxes.resize(boxCount);
	for (int i = 0; i < boxCount; ++i) {
		bboxes[i] = blas->bvh[i].bb;
	}

	// determine bounding box levels ()
	level.resize(boxCount * 2, 0);

	// traverse all level
	struct Info
	{
		scene::accel_struct::BvhNode node;
		int level;
	};

	std::stack<Info> levelStack;
	maxLevel = 0;
	levelStack.push({blas->bvh[0], 0});
	levelStack.push({blas->bvh[1], 0});
	while(!levelStack.empty())
	{
		auto e = levelStack.top();
		levelStack.pop();
		maxLevel = std::max(maxLevel, e.level);

		// push children
		auto nextIdx = e.node.index * 2;
		if (e.node.index >= blas->numInternalNodes) continue;

		levelStack.push({ blas->bvh[nextIdx], e.level + 1 });
		levelStack.push({ blas->bvh[nextIdx + 1], e.level + 1 });
		// two level indices per box because bounding box edge points count as one primitve for the box pipeline
		level[nextIdx * 2] = e.level + 1;
		level[nextIdx * 2 + 1] = e.level + 1;
		level[nextIdx * 2 + 2] = e.level + 1;
		level[nextIdx * 2 + 3] = e.level + 1;
	}

	// remove boxes for level filter
	if (m_minLevel > 0 || m_maxLevel < maxLevel)
	{
		// filter out some boxes
		int dst = 0;
		for(int src = 0; src < boxCount; ++src)
		{
			const auto lvl = level[2 * src];
			if(lvl < m_minLevel || lvl > m_maxLevel) continue; // skip data
			// keep data
			bboxes[dst] = bboxes[src];
			level[dst * 2] = level[src * 2];
			level[dst * 2 + 1] = level[src * 2];
			++dst;
		}

		// resize buffer
		bboxes.resize(dst);
		level.resize(dst * 2);
		boxCount = dst;

		if (!boxCount) return;
	}

	glGenBuffers(1, &dstBoxBuffer);
	gl::bindBuffer(gl::BufferType::ShaderStorage, dstBoxBuffer);
	gl::bufferStorage(dstBoxBuffer, bboxes.size() * sizeof(bboxes[0]), bboxes.data(), gl::StorageFlags::None);

	glGenBuffers(1, &dstLevelBuffer);
	gl::bindBuffer(gl::BufferType::ShaderStorage, dstLevelBuffer);
	gl::bufferStorage(dstLevelBuffer, level.size() * sizeof(level[0]), level.data(), gl::StorageFlags::None);
}
