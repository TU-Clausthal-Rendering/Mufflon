#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon::renderer {

struct PForwardDummy {
	PARAM_ENUM(dummy = Values::Dummy2, Dummy1= 4, Dummy2 = 8);
	static constexpr ParamDesc get_desc() noexcept {
		return { "dummy", ParameterTypes::ENUM };
	}
};
	
using ForwardParameters = ParameterHandler<PForwardDummy>;

} // namespace mufflon::renderer