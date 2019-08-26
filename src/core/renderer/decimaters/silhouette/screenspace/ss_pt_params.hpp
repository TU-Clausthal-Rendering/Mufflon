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

struct RadianceTarget {
	static constexpr const char NAME[] = "Radiance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
	static constexpr bool REQUIRED = true;
};
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
struct PenumbraTarget {
	static constexpr const char NAME[] = "Penumbra";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};
struct RadianceTransitionTarget {
	static constexpr const char NAME[] = "Radiance transition";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};
struct NumShadowPixelsTarget {
	static constexpr const char NAME[] = "Umbra pixel count";
	using PixelType = u32;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct NumPenumbraPixelsTarget {
	static constexpr const char NAME[] = "Penumbra pixel count";
	using PixelType = u32;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct PenumbraSizeTarget {
	static constexpr const char NAME[] = "Penumbra size";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using SilhouetteTargets = TargetList<
	RadianceTarget, ImportanceTarget, PolyShareTarget,
	SilhouetteWeightTarget, PenumbraTarget, RadianceTransitionTarget
>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss