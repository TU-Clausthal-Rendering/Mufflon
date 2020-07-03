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

struct HitIdTarget {
	static constexpr const char NAME[] = "Hit ID";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 2u;
};

using PtTargets = TargetList<
	RadianceTarget, PositionTarget, DepthTarget,
	NormalTarget, UvTarget, AlbedoTarget, LightnessTarget,
	HitIdTarget
>;

}} // namespace mufflon::renderer