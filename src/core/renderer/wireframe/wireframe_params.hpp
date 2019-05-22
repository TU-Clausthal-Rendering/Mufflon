#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon {namespace renderer {

struct PWireframeLinewidth {
	float lineWidth = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Line width", ParameterTypes::FLOAT };
	}
};

struct PWireframeMaxTraceDepth {
	int maxTraceDepth = 1000;
	static ParamDesc get_desc() noexcept {
		return { "Maximum trace depth", ParameterTypes::INT };
	}
};

using WireframeParameters = ParameterHandler<PWireframeLinewidth, PWireframeMaxTraceDepth>;

}} // namespace mufflon::renderer
