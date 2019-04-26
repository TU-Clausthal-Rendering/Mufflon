#pragma once

#include "util/types.hpp"
#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"
#include "core/scene/types.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace bpm {

using SilhouetteParameters = ParameterHandler<
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PDirectIndirectRatio, PSharpnessFactor,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PMergeRadius, PProgressive
>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::bpm