#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon::renderer {

struct PForwardDummy {
	float dummy = 0.0f;
	static ParamDesc get_desc() noexcept {
		return { "dummy", ParameterTypes::FLOAT };
	}
};
	
using ForwardParameters = ParameterHandler<PForwardDummy>;

} // namespace mufflon::renderer