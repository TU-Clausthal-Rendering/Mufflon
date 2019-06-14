#pragma once

#include "core/renderer/renderer_base.hpp"
#include "core/opengl/gl_object.h"

namespace mufflon::gl {
    struct Pipeline;
}

namespace mufflon::renderer {
    
class GlRendererBase : public RendererBase<Device::OPENGL> {
public:
	enum class Attribute : uint32_t {
        None = 0,
		Position = 1,
		Normal = 1 << 1,
		Texcoord = 1 << 2,
        Material = 1 << 3,
        All = 0xFFFFFFFF
	};

	GlRendererBase(bool useDepth, bool useStencil);
	virtual ~GlRendererBase() = default;

    // reload textures with appropriate size
    void post_reset() override;
protected:
	struct CameraTransforms {
		ei::Mat4x4 viewProj;
		ei::Mat4x4 view;
		ei::Mat4x4 projection;
		ei::Mat4x4 invView;

		ei::Vec3 position;
		float near;
		ei::Vec3 direction;
		float far;
		ei::UVec2 screen;
	};

    // enable framebuffer and clear textures
	void begin_frame(ei::Vec4 clearColor);
    // copy framebuffer to rendertarget buffer
	void end_frame();

	// used uniform locations:
    // 1: instance transform
    // used shader storage bindings:
    // 0: material data
    // 1: material per primitive ids
	void draw_triangles(const gl::Pipeline& pipe, Attribute attribs);

	// used uniform locations:
    // 1: instance transform
    // used shader storage bindings:
    // 0: material data (material ids are passed as vertex attribute)
	void draw_spheres(const gl::Pipeline& pipe, Attribute attribs);

    // used uniform locations:
    // 1: instance transform
    // 2: num triangles (for material offset)
    // used shader storage bindings:
    // 0: material data
    // 1: material per primitive ids
	void draw_quads(const gl::Pipeline& pipe, Attribute attribs);

	CameraTransforms get_camera_transforms() const;

    // returns aligned division: size / alignment + !!(size % alignment)
	static size_t get_aligned(size_t size, size_t alignment);

	uint32_t m_depthStencilFormat;
	gl::Texture m_depthTarget;
	gl::Texture m_colorTargets[OutputValue::TARGET_COUNT];
	gl::Framebuffer m_framebuffer;
private:
	uint32_t m_depthAttachmentType;
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
