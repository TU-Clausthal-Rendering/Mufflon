#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PNeeMergeRadius {
	float neeMergeRadius { 0.001f };
	static ParamDesc get_desc() noexcept {
		return {"NEE merge radius", ParameterTypes::FLOAT};
	}
};

struct PTargetFlux {
	float targetFlux { 0.0005f };
	static ParamDesc get_desc() noexcept {
		return {"Target flux", ParameterTypes::FLOAT};
	}
};

struct PSecondaryNEEs {
	bool secondaryNEEs { true };
	static ParamDesc get_desc() noexcept {
		return {"Secondary NEEs", ParameterTypes::BOOL};
	}
};

using NebParameters = ParameterHandler<
	PMinPathLength,
	PMaxPathLength,
//	PNeeCount,
	PNeePositionGuide,
	PMergeRadius,
	PNeeMergeRadius,
	PTargetFlux,
	PSecondaryNEEs
>;

}} // namespace mufflon::renderer