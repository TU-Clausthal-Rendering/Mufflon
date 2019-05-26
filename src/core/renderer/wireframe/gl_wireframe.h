#pragma once

#include "wireframe_params.hpp"
#include "core/renderer/gl_renderer_base.h"
#include "core/opengl/gl_pipeline.h"

namespace mufflon::renderer {
    
class GlWireframe final : public GlRendererBase {
public:
	GlWireframe();
	~GlWireframe() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Wireframe"; }
	StringView get_short_name() const noexcept final { return "WF"; }

	void on_reset() override;

private:
	WireframeParameters m_params = {};
	ei::Mat4x4 m_viewProjMatrix;
	gl::Program m_program;

	gl::Pipeline m_pipe;
	gl::VertexArray m_vao;
};
}
