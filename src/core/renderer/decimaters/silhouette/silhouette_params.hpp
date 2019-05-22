#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette {

struct PImportanceIterations {
	int importanceIterations{ 100 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PTargetReduction {
	float reduction{ 0.9f };
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
	int decimationIterations{ 1 };
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

struct PSharpnessFactor {
	float sharpnessFactor = 10.f;
	static ParamDesc get_desc() noexcept {
		return { "2/(1+exp(-BxDF/factor)) - 1", ParameterTypes::FLOAT };
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

struct PShadowWeight {
	float shadowWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow paths", ParameterTypes::FLOAT };
	}
};

struct PShadowSilhouetteWeight {
	float shadowSilhouetteWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow silhouette paths", ParameterTypes::FLOAT };
	}
};

}}}} // namespace mufflon::renderer::decimaters::silhouette