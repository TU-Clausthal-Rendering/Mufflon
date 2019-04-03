#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon::renderer::decimaters::importance {

struct PDecimationIterations {
	int decimationIterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PSharpnessFactor {
	float sharpnessFactor = 10.f;
	static ParamDesc get_desc() noexcept {
		return { "2/(1+exp(-BxDF/factor)) - 1", ParameterTypes::FLOAT };
	}
};

struct PMaxNormalDeviation {
	float maxNormalDeviation = 60.f;
	static ParamDesc get_desc() noexcept {
		return { "Max. normal deviation from collapse", ParameterTypes::FLOAT };
	}
};

struct PViewWeight {
	float viewWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of view paths", ParameterTypes::FLOAT };
	}
};

struct PLightWeight {
	float lightWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of light paths", ParameterTypes::FLOAT };
	}
};

struct PRenderUpdate {
	bool renderUpdate = false;
	static ParamDesc get_desc() noexcept {
		return { "Show update between decimations", ParameterTypes::BOOL };
	}
};

using ImportanceParameters = ParameterHandler<
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PSharpnessFactor, PMaxNormalDeviation,
	PViewWeight, PLightWeight,
	PMaxPathLength,
	PRenderUpdate
>;

} // namespace mufflon::renderer::decimaters::importance