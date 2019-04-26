#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon::renderer::decimaters::importance {

struct PImportanceIterations {
	int importanceIterations{ 1 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PTargetReduction {
	float reduction{ 0.875f };
	static ParamDesc get_desc() noexcept {
		return { "Target mesh reduction", ParameterTypes::FLOAT };
	}
};

struct PInitialReduction {
	float initialReduction = 0.f;
	static ParamDesc get_desc() noexcept {
		return { "Reduce mesh initially", ParameterTypes::FLOAT };
	}
};

struct PVertexThreshold {
	int threshold{ 100 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation threshold", ParameterTypes::INT };
	}
};

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
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide,
	PRenderUpdate
>;

} // namespace mufflon::renderer::decimaters::importance