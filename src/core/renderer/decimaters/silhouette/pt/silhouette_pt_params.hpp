#pragma once

#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

using SilhouetteParameters = ParameterHandler<
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PDirectIndirectRatio, PSharpnessFactor,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide
>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt