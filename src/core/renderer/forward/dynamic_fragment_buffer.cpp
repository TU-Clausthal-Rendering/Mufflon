#include "dynamic_fragment_buffer.hpp"
#include "glad/glad.h"
#include "util/assert.hpp"

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
	gl::bufferStorage(m_fragmentCounts, sizeof(uint32_t) *  alignPowerOfTwo(m_curScanSize, 4), nullptr, gl::StorageFlags::None);
	// clear buffer with 0
	uint32_t zero = 0;
	gl::clearBufferData(m_fragmentCounts, alignPowerOfTwo(m_curScanSize, 4), &zero);


	// auxiliary buffers for scan
	m_auxBuffer.clear();
	uint32_t bs = m_curScanSize;
	while(bs > 1)
	{
		m_auxBuffer.emplace_back(gl::genBuffer());
		gl::bufferStorage(m_auxBuffer.back(), sizeof(uint32_t) * alignPowerOfTwo(bs, 4), nullptr, gl::StorageFlags::None);
		bs /= alignment;
	}

	// staging buffer for fragment count
	if(!m_stageBuffer)
	{
		glGenBuffers(1, &m_stageBuffer);
		gl::bufferStorage(m_stageBuffer, sizeof(uint32_t), nullptr, gl::StorageFlags::ClientStorage);
	}

	// TODO store width in sort blend shader
}

void DynamicFragmentBuffer::bindCountBuffer() {
	mAssert(m_fragmentCounts);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_fragmentCounts);
}

void DynamicFragmentBuffer::prepareFragmentBuffer()
{
}

void DynamicFragmentBuffer::blendFragmentBuffer()
{
}
}
