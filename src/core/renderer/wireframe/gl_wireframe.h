#pragma once

#include "wireframe_params.hpp"
#include "core/renderer/gl_renderer_base.h"

namespace mufflon::renderer {
    
class GlWireframe final : public GlRendererBase {
public:
	GlWireframe() = default;
	~GlWireframe() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Wireframe"; }
	StringView get_short_name() const noexcept final { return "WF"; }

	void on_reset() override;

private:
	WireframeParameters m_params = {};
};
}