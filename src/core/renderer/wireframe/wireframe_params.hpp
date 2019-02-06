#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon {namespace renderer {

struct PWireframeThickness {
	float thickness{ 0.025f };
	static ParamDesc get_desc() noexcept {
		return { "Border thickness", ParameterTypes::FLOAT };
	}
};

struct PWireframeNormalize {
	bool normalize = false;
	static ParamDesc get_desc() noexcept {
		return { "Normalize thickness", ParameterTypes::BOOL };
	}
};

using WireframeParameters = ParameterHandler<PWireframeThickness, PWireframeNormalize>;

}} // namespace mufflon::renderer