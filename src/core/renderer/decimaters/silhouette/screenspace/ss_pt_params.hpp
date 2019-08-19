#pragma once

#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

using SilhouetteParameters = ParameterHandler<
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PSelectiveImportance, PShadowSizeWeight,
	PDirectIndirectRatio, PSharpnessFactor,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide
>;

struct ShadowTarget {
	static constexpr const char NAME[] = "Shadow Areas";
	using PixelType = u32;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct SilhouetteTarget {
	static constexpr const char NAME[] = "Shadow Silhouette";
	using PixelType = u32;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct SilhouetteWeightTarget {
	static constexpr const char NAME[] = "Silhouette weight";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using SilhouetteTargets = TargetList<
	ImportanceTarget, ShadowTarget,
	SilhouetteTarget, SilhouetteWeightTarget
>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss