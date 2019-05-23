#pragma once

#include "forward_params.hpp"
#include "core/renderer/renderer_base.hpp"

namespace mufflon::renderer {
	
class GlForward final : public RendererBase<Device::OPENGL> {
public:
	// Initialize all resources required by this renderer
	GlForward() = default;
	~GlForward() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Forward"; }
	StringView get_short_name() const noexcept final { return "FW"; }
	void post_descriptor_requery() final;

private:
	ForwardParameters m_params = {};
};


}