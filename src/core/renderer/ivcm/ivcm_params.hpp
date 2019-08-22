#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PShowDensity {
	bool showDensity { false };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Show Density", ParameterTypes::BOOL};
	}
};

struct PHeuristic {
	PARAM_ENUM(heuristic = Values::VCM, VCM, VCMPlus, VCMStar, IVCM);
	static constexpr ParamDesc get_desc() noexcept {
		return { "Heuristic", ParameterTypes::ENUM };
	}
};

using IvcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive,
	PShowDensity,
	PHeuristic
>;

}} // namespace mufflon::renderer