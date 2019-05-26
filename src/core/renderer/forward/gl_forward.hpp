#pragma once

#include "forward_params.hpp"
#include "core/renderer/renderer_base.hpp"

namespace mufflon::renderer {
	
class GlForward final : public RendererBase<Device::OPENGL> {
public:
	// Initialize all resources required by this renderer
	GlForward() = default;
	~GlForward() { unload(); }

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Forward"; }
	StringView get_short_name() const noexcept final { return "FW"; }

	void on_descriptor_requery() final;
    void on_reset() override;

private:
	void unload();

	ForwardParameters m_params = {};
	// render targets
    gl::Handle m_depthTarget;
	gl::Handle m_colorTarget;

	gl::Handle m_framebuffer;

	gl::Handle m_copyShader;
};


}