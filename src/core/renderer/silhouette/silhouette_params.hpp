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

struct PMemoryConstraint {
	int memoryConstraint = 100'000;
	static ParamDesc get_desc() noexcept {
		return { "Memory available for geometry", ParameterTypes::INT };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PVertexThreshold,
	PDirectIndirectRatio, PDecimationEnabled, PMemoryConstraint, PMaxPathLength>;

}} // namespace mufflon::renderer