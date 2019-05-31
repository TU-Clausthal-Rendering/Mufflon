#pragma once

#include "core/renderer/renderer_base.hpp"
#include "core/opengl/gl_object.h"

namespace mufflon::gl {
    struct Pipeline;
}

namespace mufflon::renderer {
    
class GlRendererBase : public RendererBase<Device::OPENGL> {
public:
	enum class Attribute {
        None = 0,
		Position = 1,
		Normal = 1 << 1,
		Texcoord = 1 << 2
	};

	GlRendererBase();
	virtual ~GlRendererBase() = default;

    // reload textures with appropriate size
    void on_reset() override;
protected:
	struct CameraTransforms {
		ei::Mat4x4 viewProj;
		ei::Mat4x4 view;
		ei::Mat4x4 projection;
	};

    // enable framebuffer and clear textures
	void begin_frame(ei::Vec4 clearColor);
    // copy framebuffer to rendertarget buffer
	void end_frame();

	void draw_triangles(const gl::Pipeline& pipe, Attribute attribs);
	void draw_spheres(const gl::Pipeline& pipe);
	void draw_quads(const gl::Pipeline& pipe, Attribute attribs);

	CameraTransforms get_camera_transforms() const;

    // returns aligned division: size / alignment + !!(size % alignment)
	static size_t get_aligned(size_t size, size_t alignment);

	gl::Texture m_depthTarget;
	gl::Texture m_colorTarget;
	gl::Framebuffer m_framebuffer;
private:
	gl::Program m_copyShader;
	static const size_t WORK_GROUP_SIZE = 16;
};

inline GlRendererBase::Attribute operator|(GlRendererBase::Attribute l, GlRendererBase::Attribute r) {
    return GlRendererBase::Attribute(uint32_t(l) | uint32_t(r));
}
inline bool operator&(GlRendererBase::Attribute l, GlRendererBase::Attribute r) {
    return (uint32_t(l) & uint32_t(r)) != 0;
}
}
