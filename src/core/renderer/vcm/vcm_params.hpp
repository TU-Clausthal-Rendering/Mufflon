#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

using VcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive
>;

using VcmTargets = TargetList<
	RadianceTarget, PositionTarget,
	NormalTarget, AlbedoTarget, LightnessTarget
>;

}} // namespace mufflon::renderer