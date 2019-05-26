#pragma once

#include "core/renderer/renderer_base.hpp"
#include "core/opengl/gl_object.h"

namespace mufflon::renderer {
    
class GlRendererBase : public RendererBase<Device::OPENGL> {
public:
	GlRendererBase();
	virtual ~GlRendererBase() = default;

    // reload textures with appropriate size
    void on_reset() override;
protected:
    // enable framebuffer and clear textures
	void begin_frame(ei::Vec4 clearColor);
    // copy framebuffer to rendertarget buffer
	void end_frame();

    // returns aligned division: size / alignment + !!(size % alignment)
	static size_t get_aligned(size_t size, size_t alignment);

	gl::Texture m_depthTarget;
	gl::Texture m_colorTarget;
	gl::Framebuffer m_framebuffer;
private:
	gl::Program m_copyShader;
	static const size_t WORK_GROUP_SIZE = 16;
};
}