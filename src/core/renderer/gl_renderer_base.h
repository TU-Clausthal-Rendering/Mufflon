#pragma once

#include "core/cameras/pinhole.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/opengl/gl_object.h"

namespace mufflon::gl {
    struct Pipeline;
}

namespace mufflon::renderer {

class GlRenderer {
public:
	enum class Attribute : uint32_t {
		None = 0,
		Position = 1,
		Normal = 1 << 1,
		Texcoord = 1 << 2,
        Material = 1 << 3,
        Light = 1 << 4,
        All = 0xFFFFFFFF
	};

protected:
	GlRenderer(const u32 colorAttachmentCount, bool useDepth, bool useStencil);
	virtual ~GlRenderer() = default;

	// reload textures with appropriate size
	void reset(int width, int height);
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

	virtual scene::SceneDescriptor<Device::OPENGL> get_scene_descriptor() = 0;
	virtual std::optional<ConstRenderTargetBuffer<Device::OPENGL, char>> get_target(const u32 index) = 0;

	// enable framebuffer and clear textures
	void begin_frame(ei::Vec4 clearColor, int width, int height);
	// copy framebuffer to rendertarget buffer
	void end_frame(int width, int height);

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

	virtual CameraTransforms get_camera_transforms() const = 0;

	// returns aligned division: size / alignment + !!(size % alignment)
	static size_t get_aligned(size_t size, size_t alignment);

	uint32_t m_depthStencilFormat;
	gl::Texture m_depthTarget;
	std::vector<gl::Texture> m_colorTargets;
	gl::Framebuffer m_framebuffer;
private:
    // bind some static attributes (light, material)
	void bindStaticAttribs(const gl::Pipeline& pipe, Attribute attribs);

	uint32_t m_depthAttachmentType;
	gl::Program m_copyShader;
	static const size_t WORK_GROUP_SIZE = 16;
};

inline GlRenderer::Attribute operator|(GlRenderer::Attribute l, GlRenderer::Attribute r) {
	return GlRenderer::Attribute(uint32_t(l) | uint32_t(r));
}
inline bool operator&(GlRenderer::Attribute l, GlRenderer::Attribute r) {
	return (uint32_t(l) & uint32_t(r)) != 0;
}
   
template < class TL >
class GlRendererBase : public RendererBase<Device::OPENGL, TL>, protected GlRenderer {
public:
	GlRendererBase(bool useDepth, bool useStencil) :
		GlRenderer(TL::TARGET_COUNT, useDepth, useStencil)
	{}
	virtual ~GlRendererBase() = default;

    // reload textures with appropriate size
	void post_reset() override {
		this->reset(this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
	}
protected:
	static constexpr u32 COLOR_ATTACHMENTS = TL::TARGET_COUNT;

    // enable framebuffer and clear textures
	void begin_frame(ei::Vec4 clearColor) {
		GlRenderer::begin_frame(clearColor, this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
	}
    // copy framebuffer to rendertarget buffer
	void end_frame() {
		GlRenderer::end_frame(this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
	}
	
	scene::SceneDescriptor<Device::OPENGL> get_scene_descriptor() override {
		return this->m_sceneDesc;
	}
	virtual std::optional<ConstRenderTargetBuffer<Device::OPENGL, char>> get_target(const u32 index) override {
		return this->m_outputBuffer.get_target(index);
	}

	CameraTransforms get_camera_transforms() const override {
		CameraTransforms t;

		auto* cam = this->m_currentScene->get_camera();

		t.position = cam->get_position(0);
		t.direction = cam->get_view_dir(0);

		t.near = cam->get_near();
		t.far = cam->get_far();
		t.screen.x = this->m_outputBuffer.get_width();
		t.screen.y = this->m_outputBuffer.get_height();

		float fov = 1.5f;
		if(auto pcam = dynamic_cast<const cameras::Pinhole*>(cam)) {
			fov = pcam->get_vertical_fov();
		}
		t.projection = ei::perspectiveGL(fov,
										 float(t.screen.x) / t.screen.y,
										 t.near, t.far);
		t.view = ei::camera(
			t.position,
			t.position + t.direction,
			cam->get_up_dir(0)
		);
		t.invView = ei::invert(t.view);
		t.viewProj = t.projection * t.view;
		// transpose since opengl expects column major
		t.projection = ei::transpose(t.projection);
		t.view = ei::transpose(t.view);
		t.viewProj = ei::transpose(t.viewProj);
		t.invView = ei::transpose(t.invView);

		return t;
	}
};

} // namespace mufflon::renderer