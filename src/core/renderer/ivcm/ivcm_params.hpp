#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

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
	PHeuristic
>;

using IvcmTargets = TargetList<
	RadianceTarget, PositionTarget,
	NormalTarget, AlbedoTarget, LightnessTarget,
	DensityTarget
>;

}} // namespace mufflon::renderer