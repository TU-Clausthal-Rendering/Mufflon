#pragma once
#include "gl_pipeline.hpp"

namespace mufflon::gl {
class Context {
public:
	static void set(const Pipeline& pipeline);
private:
	Context();
	static Context& get();

	struct State {
		RasterizerState rasterizer;
		DepthStencilState depthStencil;
		BlendState blend;
		PatchState patch;
        // sampler might get added later
	} m_state;
	gl::Handle m_emptyVao;
};
}