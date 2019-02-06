#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PImportanceIterations {
	int iterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations>;

}} // namespace mufflon::renderer