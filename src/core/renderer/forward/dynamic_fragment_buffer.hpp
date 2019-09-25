#pragma once

#include "core/opengl/gl_object.h"
#include "core/opengl/gl_pipeline.h"
#include <vector>

namespace mufflon::renderer {
	
	// usage:
	// 1. draw opaque geometry into framebuffer
	// 2. bindCountBuffer() + count transparent silhouette (count fragments per pixel)
	// 3. prepareFragmentBuffer() (prepares fragment buffer + resize buffer)
	// 4. draw transparent geometry and store in fragment buffer
	// 5. blendFragmentBuffer() blends transparent and color target
	class DynamicFragmentBuffer
	{
	public:
		void init(const gl::Framebuffer& framebuffer, int width, int height);

		void bindCountBuffer();

		void prepareFragmentBuffer();

		void blendFragmentBuffer();
	private:
		gl::Buffer m_fragmentBuffer;

		gl::Buffer m_fragmentCounts;
		gl::Buffer m_stageBuffer;

		gl::Pipeline m_countPipeline;
		gl::Pipeline m_blendPipeline;

		gl::Program m_scanProgram;
		gl::Program m_blendProgram;

		uint32_t m_curScanSize = 0;
		uint32_t m_curLastIndex = 0;

		std::vector<gl::Buffer> m_auxBuffer;
	};
}
