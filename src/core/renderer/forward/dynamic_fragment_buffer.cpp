#include "dynamic_fragment_buffer.hpp"
#include "glad/glad.h"
#include "util/assert.hpp"
#include "core/opengl/program_builder.h"
#include "core/opengl/gl_context.h"

static const int WORKGROUP_SIZE = 1024;
static const int ELEM_PER_THREAD_SCAN = 8;


inline uint32_t alignPowerOfTwo(uint32_t size, uint32_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
}

namespace mufflon::renderer {

void DynamicFragmentBuffer::init(const gl::Framebuffer& framebuffer, int width, int height) {
	// determine alignment that is required by the scan algorithm
	uint32_t alignment = WORKGROUP_SIZE * ELEM_PER_THREAD_SCAN;
	m_curScanSize = alignPowerOfTwo(width * height, alignment);
	m_curLastIndex = width * height - 1;

	// setup fragment count buffer
	glGenBuffers(1, &m_fragmentCounts);
	gl::bindBuffer(gl::BufferType::ShaderStorage, m_fragmentCounts);
	gl::bufferStorage(m_fragmentCounts, sizeof(uint32_t) *  alignPowerOfTwo(m_curScanSize, 4), nullptr, gl::StorageFlags::DynamicStorage);
	// clear buffer with 0
	uint32_t zero = 0;
	gl::clearBufferData(m_fragmentCounts, alignPowerOfTwo(m_curScanSize, 4), &zero);


	// auxiliary buffers for scan
	m_auxBuffer.clear();
	uint32_t bs = m_curScanSize;
	while(bs > 1)
	{
		m_auxBuffer.emplace_back(gl::genBuffer());
		gl::bindBuffer(gl::BufferType::ShaderStorage, m_auxBuffer.back());
		auto numElements = alignPowerOfTwo(bs, 4);
		gl::bufferStorage(m_auxBuffer.back(), sizeof(uint32_t) * numElements, nullptr, gl::StorageFlags::None);
		m_numAuxBufferElements.emplace_back(numElements);
		bs /= alignment;
	}

	// staging buffer for fragment count
	if(!m_stageBuffer)
	{
		glGenBuffers(1, &m_stageBuffer);
		gl::bindBuffer(gl::BufferType::ShaderStorage, m_stageBuffer);
		gl::bufferStorage(m_stageBuffer, sizeof(uint32_t), nullptr, gl::StorageFlags::ClientStorage);
	}

	// create fragment buffer storage
	glGenBuffers(1, &m_fragmentBuffer);
	gl::bindBuffer(gl::BufferType::ShaderStorage, m_fragmentBuffer);
	gl::bufferStorage(m_fragmentBuffer, 8 * m_numFragments, nullptr, gl::StorageFlags::DynamicStorage);

	m_scanProgram = gl::ProgramBuilder()
		.add_file("shader/scan.glsl")
		.build_shader(gl::ShaderType::Compute)
		.build_program();

	m_scanPushProgram = gl::ProgramBuilder()
		.add_file("shader/scanpush.glsl")
		.build_shader(gl::ShaderType::Compute)
		.build_program();

	m_blendProgram = gl::ProgramBuilder()
		.add_file("shader/quad.glsl", false)
		.build_shader(gl::ShaderType::Vertex)
		.add_file("shader/sort_fragments.glsl")
		.build_shader(gl::ShaderType::Fragment)
		.build_program();

	// set screen width
	glProgramUniform1ui(m_blendProgram, 0, width);

	m_blendPipeline.framebuffer = framebuffer;
	m_blendPipeline.program = m_blendProgram;
	m_blendPipeline.depthStencil.depthTest = false;
	m_blendPipeline.depthStencil.depthWrite = false;
	m_blendPipeline.rasterizer.cullMode = gl::CullMode::None;
	m_blendPipeline.blend.enableBlending = gl::BlendMode::Blend;
	m_blendPipeline.blend.renderTarget[0].colorBlendOp = gl::BlendOp::Add;
	m_blendPipeline.blend.renderTarget[0].srcColorFactor = gl::BlendFactor::One;
	m_blendPipeline.blend.renderTarget[0].dstColorFactor = gl::BlendFactor::SrcAlpha;
}

void DynamicFragmentBuffer::bindCountBuffer() {
	// buffer is cleared in the scan shader
	//uint32_t zero = 0;
	//gl::clearBufferData(m_fragmentCounts, alignPowerOfTwo(m_curScanSize, 4), &zero);

	mAssert(m_fragmentCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_fragmentCounts);

	glMemoryBarrier(GL_SHADER_STORAGE_BUFFER);
}

void DynamicFragmentBuffer::prepareFragmentBuffer()
{
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	doScan();

	// resize fragment buffer if required
	uint32_t numFragments = 0;
	glGetNamedBufferSubData(m_stageBuffer, 0, sizeof(numFragments), &numFragments);
	if(numFragments > m_numFragments)
	{
		// resize fragment buffer
		m_numFragments = numFragments;
		glGenBuffers(1, &m_fragmentBuffer);
		gl::bindBuffer(gl::BufferType::ShaderStorage, m_fragmentBuffer);
		gl::bufferStorage(m_fragmentBuffer, 8 * m_numFragments, nullptr, gl::StorageFlags::DynamicStorage);
	}

	// bind resources for color pass
	mAssert(m_fragmentCounts);
	// counter (will be counted down to 0's)
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_fragmentCounts);
	// base buffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, m_auxBuffer.front());
	// storage
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m_fragmentBuffer);
}

void DynamicFragmentBuffer::blendFragmentBuffer()
{
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	gl::Context::set(m_blendPipeline);

	// draw fullscreen quad
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void DynamicFragmentBuffer::doScan()
{
	glUseProgram(m_scanProgram);

	auto bs = m_curScanSize; int i = 0;
	const auto elemPerWk = WORKGROUP_SIZE * ELEM_PER_THREAD_SCAN;
	// Hierarchical scan of blocks
	while (bs > 1)
	{
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		if (i == 0) // in the first step the counting buffer should be used
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_fragmentCounts);
		else
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_auxBuffer.at(i));

		// shader storage buffer binding
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_auxBuffer.at(i));

		// Bind the auxiliary buffer for the next step or unbind (in the last step)
		if (i + 1 < m_auxBuffer.size())
			// shader storage buffer binding
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_auxBuffer.at(i + 1));
		else glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);

		glUniform1ui(0, m_numAuxBufferElements.at(i));
		//glUniform1ui(0, 0);
		glDispatchCompute((bs + elemPerWk - 1) / elemPerWk, 1, 1);

		bs /= elemPerWk;
		++i;
	}

	// Complete Intra-block scan by pushing the values up
	glUseProgram(m_scanPushProgram);
	glUniform1ui(0, elemPerWk);
	glUniform1ui(1, 0);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_stageBuffer);

	--i; bs = m_curScanSize;
	while (bs > elemPerWk) bs /= elemPerWk;
	while (bs < m_curScanSize)
	{
		bs *= elemPerWk;

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		// bind as shader storage
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_auxBuffer.at(i - 1));
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_auxBuffer.at(i));

		if (i == 1) // last write
			glUniform1ui(1, m_curLastIndex);

		glDispatchCompute((bs - elemPerWk) / 64, 1, 1);
		--i;
	}
}
}
