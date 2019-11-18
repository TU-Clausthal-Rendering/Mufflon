#pragma once

#include "core/renderer/decimaters/animation/animation_params.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace animation { namespace pt {

using SilhouetteParameters = ParameterHandler <
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PSelectiveImportance, PShadowSizeWeight,
	PDirectIndirectRatio, PSharpnessFactor,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide,
	PSlidingWindow, PVertexDistMethod, PImpWeightMethod
>;

struct PShadowOmitted {
	static constexpr const char NAME[] = "Shadow Omitted";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

struct PShadowRecorded {
	static constexpr const char NAME[] = "Shadow Recorded";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};


using SilhouetteTargets = TargetList<ImportanceTarget, PolyShareTarget, PShadowRecorded, PShadowOmitted>;

}}}}} // namespace mufflon::renderer::decimaters::animation::pt