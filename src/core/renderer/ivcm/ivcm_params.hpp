#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PShowDensity {
	bool showDensity { false };
	static ParamDesc get_desc() noexcept {
		return {"Show Density", ParameterTypes::BOOL};
	}
};

using IvcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive,
	PShowDensity
>;

}} // namespace mufflon::renderer