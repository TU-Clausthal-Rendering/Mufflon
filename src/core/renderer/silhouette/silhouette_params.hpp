#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PDecimationIterations {
	int decimationIterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PDirectIndirectRatio {
	float directIndirectRatio{ 0.02f };
	static ParamDesc get_desc() noexcept {
		return { "Ratio threshold for direct/indirect illumination", ParameterTypes::FLOAT };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PVertexThreshold,
	PDirectIndirectRatio, PDecimationEnabled, PMaxPathLength>;

}} // namespace mufflon::renderer