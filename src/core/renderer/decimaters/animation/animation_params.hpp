#pragma once

#include "core/renderer/parameter.hpp"
#include <tuple>

namespace mufflon { namespace renderer { namespace decimaters { namespace animation {

struct PSelectiveImportance {
	PARAM_ENUM(impSelection, ALL, VIEW, DIRECT, INDIRECT, SILHOUETTE) = Values::ALL;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Selective importance", ParameterTypes::ENUM };
	}

	CUDA_FUNCTION bool show_view() const noexcept { return impSelection == Values::ALL || impSelection == Values::VIEW; }
	CUDA_FUNCTION bool show_direct() const noexcept { return impSelection == Values::ALL || impSelection == Values::DIRECT; }
	CUDA_FUNCTION bool show_indirect() const noexcept { return impSelection == Values::ALL || impSelection == Values::INDIRECT; }
	CUDA_FUNCTION bool show_silhouette() const noexcept { return impSelection == Values::ALL || impSelection == Values::SILHOUETTE; }
};

struct PSlidingWindow {
	int slidingWindowHalfWidth{ 2 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Sliding window half-width", ParameterTypes::INT };
	}
};

struct PShadowSizeWeight {
	PARAM_ENUM(shadowSizeWeight, INVERSE, INVERSE_SQR, INVERSE_EXP) = Values::INVERSE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Shadow size weight", ParameterTypes::ENUM };
	}
};

struct PImportanceIterations {
	int importanceIterations{ 1 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PTargetReduction {
	float reduction{ 0.f };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Target mesh reduction", ParameterTypes::FLOAT };
	}
};

struct PInitialReduction {
	float initialReduction = 0.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Reduce mesh initially", ParameterTypes::FLOAT };
	}
};

struct PVertexThreshold {
	int threshold{ 100 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Decimation threshold", ParameterTypes::INT };
	}
};

struct PDecimationIterations {
	int decimationIterations{ 1 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PDirectIndirectRatio {
	float directIndirectRatio{ 0.02f };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Ratio threshold for direct/indirect illumination", ParameterTypes::FLOAT };
	}
};

struct PSharpnessFactor {
	float sharpnessFactor = 10.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "2/(1+exp(-BxDF/factor)) - 1", ParameterTypes::FLOAT };
	}
};

struct PViewWeight {
	float viewWeight = 1.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Imp. weight of view paths", ParameterTypes::FLOAT };
	}
};

struct PLightWeight {
	float lightWeight = 1.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Imp. weight of light paths", ParameterTypes::FLOAT };
	}
};

struct PShadowWeight {
	float shadowWeight = 1.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow paths", ParameterTypes::FLOAT };
	}
};

struct PShadowSilhouetteWeight {
	float shadowSilhouetteWeight = 1.f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow silhouette paths", ParameterTypes::FLOAT };
	}
};
struct PVertexDistMethod {
	PARAM_ENUM(vertexDistMethod, AVERAGE, MAX) = Values::AVERAGE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Vertex distribution method", ParameterTypes::ENUM };
	}
};
struct PImpWeightMethod {
	PARAM_ENUM(impWeightMethod, AVERAGE, MAX) = Values::AVERAGE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance weighting method", ParameterTypes::ENUM };
	}
};


struct ImportanceTarget {
	static constexpr const char NAME[] = "Importance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct PolyShareTarget {
	static constexpr const char NAME[] = "ShadowSum";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

}}}} // namespace mufflon::renderer::decimaters::animation