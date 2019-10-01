#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

using BptParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength
>;

using BptTargets = TargetList<
	RadianceTarget, LightnessTarget
>;

}} // namespace mufflon::renderer