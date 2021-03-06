#pragma once

#include "forward_params.hpp"
#include "core/opengl/gl_object.hpp"
#include "core/renderer/gl_renderer_base.hpp"
#include "core/opengl/gl_pipeline.hpp"

namespace mufflon::scene::textures {
	class Texture;
}

namespace mufflon::renderer {
	
class GlForward final : public GlRendererBase<ForwardTargets> {
public:
	// Initialize all resources required by this renderer
	GlForward(mufflon::scene::WorldContainer& world, void* mufflonInstanceHdl);
	~GlForward() override = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Forward"; }
	static constexpr StringView get_short_name_static() noexcept { return "FW"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

    void post_reset() override;
private:
	void init();

	void* m_mufflonInstanceHdl;
	ForwardParameters m_params = {};

	gl::Program m_triangleProgram;
	gl::Program m_quadProgram;
	gl::Program m_sphereProgram;

	gl::Pipeline m_trianglePipe;
	gl::Pipeline m_quadPipe;
	gl::Pipeline m_spherePipe;

	gl::VertexArray m_triangleVao;
	gl::VertexArray m_spheresVao;

	gl::Buffer m_transformBuffer;
};


}
