#pragma once

#include "wireframe_params.hpp"
#include "core/renderer/gl_renderer_base.hpp"
#include "core/opengl/gl_pipeline.hpp"

namespace mufflon::renderer {

class GlWireframe final : public GlRendererBase<WireframeTargets> {
public:
	GlWireframe(mufflon::scene::WorldContainer& world);
	~GlWireframe() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Wireframe"; }
	static constexpr StringView get_short_name_static() noexcept { return "WF"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() override;

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
