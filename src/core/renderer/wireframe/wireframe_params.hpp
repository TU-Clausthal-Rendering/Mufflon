#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon {namespace renderer {

struct PWireframeLinewidth {
	float lineWidth = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Line width", ParameterTypes::FLOAT };
	}
};

using WireframeParameters = ParameterHandler<PWireframeLinewidth>;

}} // namespace mufflon::renderer
