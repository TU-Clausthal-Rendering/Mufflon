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
	GlWireframeParameters m_params = {};
	
	gl::Program m_triangleProgram;
	gl::Program m_quadProgram;
	gl::Program m_quadDepthProgram;
	gl::Program m_sphereProgram;
	gl::Program m_sphereDepthProgram;

	gl::Pipeline m_trianglePipe;
	gl::Pipeline m_triangleDepthPipe;
	gl::Pipeline m_quadPipe;
	gl::Pipeline m_quadDepthPipe;
	gl::Pipeline m_spherePipe;
	gl::Pipeline m_sphereDepthPipe;
	
    gl::VertexArray m_triangleVao;
	gl::VertexArray m_spheresVao;

	gl::Buffer m_transformBuffer;
};
}
