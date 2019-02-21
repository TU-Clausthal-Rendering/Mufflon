#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon {
namespace renderer {

using ImportanceParameters = ParameterHandler<PImportanceIterations,  PTargetReduction, PVertexThreshold,
	PDecimationEnabled, PKeepImportance, PMaxPathLength>;

}
} // namespace mufflon::renderer