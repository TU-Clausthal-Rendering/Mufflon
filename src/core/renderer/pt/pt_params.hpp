#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

using PtParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide
>;

using PtTargets = TargetList<
	RadianceTarget, PositionTarget,
	NormalTarget, AlbedoTarget, LightnessTarget
>;

}} // namespace mufflon::renderer