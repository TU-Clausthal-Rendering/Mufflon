#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon {namespace renderer {

struct PWireframeLinewidth {
	int lineWidth = 1;
	static ParamDesc get_desc() noexcept {
		return { "Line width", ParameterTypes::INT };
	}
};

using WireframeParameters = ParameterHandler<PWireframeLinewidth>;

}} // namespace mufflon::renderer
