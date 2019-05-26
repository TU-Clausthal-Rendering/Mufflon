#pragma once
#include "gl_pipeline.h"

namespace mufflon::gl {
class Context {
public:
	static void set(const Pipeline& pipeline);
private:
	Context() = default;
	static Context& get();

	struct State {
		RasterizerState rasterizer;
		DepthStencilState depthStencil;
		BlendState blend;
        // sampler might get added later
	} m_state;
};
}