#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

using LtParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength
>;

using LtTargets = TargetList<
	RadianceTarget, LightnessTarget,
	DensityTarget
>;

}} // namespace mufflon::renderer
