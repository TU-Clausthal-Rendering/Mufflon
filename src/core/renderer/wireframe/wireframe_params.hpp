#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon {namespace renderer {



struct PWireframeLinewidth {
	float lineWidth = 1.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Line width", ParameterTypes::FLOAT };
	}
};

struct PWireframeMaxTraceDepth {
	int maxTraceDepth = 1000;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Maximum trace depth", ParameterTypes::INT };
	}
};

struct PWireframeEnableDepthTest {
	bool enableDepth = false;
    static constexpr ParamDesc get_desc() noexcept {
		return { "Depth Test", ParameterTypes::BOOL };
    }
};

using WireframeParameters = ParameterHandler<PWireframeLinewidth, PWireframeMaxTraceDepth>;
using GlWireframeParameters = ParameterHandler<PWireframeLinewidth, PWireframeEnableDepthTest>;

struct BorderTarget {
	static constexpr const char NAME[] = "Border";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};


using WireframeTargets = TargetList<BorderTarget>;

}} // namespace mufflon::renderer
