#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

struct PNeeMergeRadius {
	float neeMergeRadius { 0.001f };
	static constexpr ParamDesc get_desc() noexcept {
		return {"NEE merge radius", ParameterTypes::FLOAT};
	}
};

struct PTargetFlux {
	float targetFlux { 0.0005f };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Target flux", ParameterTypes::FLOAT};
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
	PNeeMergeRadius,
	PTargetFlux,
	PStdPhotons
>;

using NebTargets = TargetList<
	RadianceTarget, PositionTarget,
	NormalTarget, AlbedoTarget, LightnessTarget,
	DensityTarget
>;

}} // namespace mufflon::renderer