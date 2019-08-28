#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

struct PTargetFlux {
	float targetFlux { 1.0f };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Target rel. flux", ParameterTypes::FLOAT};
	}
};

struct PStdPhotons {
	bool stdPhotons { true };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Std photons", ParameterTypes::BOOL};
	}
};

using NebParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
//	PNeeCount,
	PNeePositionGuide,
	PMergeRadius,
	PTargetFlux,
	PStdPhotons
>;

using NebTargets = TargetList<
	RadianceTarget, DensityTarget
>;

}} // namespace mufflon::renderer