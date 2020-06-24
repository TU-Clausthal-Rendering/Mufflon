#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace combined {

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

struct PInitialGridRes {
	int initialGridRes = 32;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Initial grid res", ParameterTypes::INT };
	}
};

struct PVertexThreshold {
	int threshold{ 100 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Decimation threshold", ParameterTypes::INT };
	}
};

struct PLoadMaxMemory {
	int maxMemory{ 8'000 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Max. load memory (MB)", ParameterTypes::INT };
	}
};

struct PDecimationIterations {
	int decimationIterations{ 1 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
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

struct PClusterMaxDensity {
	float maxClusterDensity{ 0.0000005f };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Max clustering density", ParameterTypes::FLOAT };
	}
};
struct PInstanceMaxDensity {
	float maxInstanceDensity{ 5.f };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Max instance density", ParameterTypes::FLOAT };
	}
};
struct PImpStructCapacity {
	int impCapacity{ 1024 * 1024 * 4 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance data capacity", ParameterTypes::INT };
	}
};
struct PImpSumStrat {
	PARAM_ENUM(impSumStrat, NORMAL, CURV_AREA) = Values::NORMAL;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance sum strategy", ParameterTypes::ENUM };
	}
};
struct PVertexDistMethod {
	PARAM_ENUM(vertexDistMethod, AVERAGE, MAX, AVERAGE_ALL, MAX_ALL) = Values::AVERAGE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Vertex distribution method", ParameterTypes::ENUM };
	}
};
struct PImpWeightMethod {
	PARAM_ENUM(impWeightMethod, AVERAGE, MAX, AVERAGE_ALL, MAX_ALL) = Values::AVERAGE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance weighting method", ParameterTypes::ENUM };
	}
};
struct PSlidingWindow {
	int slidingWindowHalfWidth{ 2 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Sliding window half-width", ParameterTypes::INT };
	}
};

using CombinedParameters = ParameterHandler <
	PImportanceIterations, PTargetReduction,
	PInitialReduction, PInitialGridRes, PVertexThreshold, PLoadMaxMemory,
	PSelectiveImportance, PImpSumStrat,
	PClusterMaxDensity, PInstanceMaxDensity, PImpStructCapacity,
	PVertexDistMethod, PImpWeightMethod,
	PViewWeight, PLightWeight,
	PShadowWeight, PShadowSilhouetteWeight,
	PSlidingWindow,
	PMaxPathLength, PNeeCount
>;

struct ImportanceTarget {
	static constexpr const char NAME[] = "Importance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct PenumbraTarget {
	static constexpr const char NAME[] = "Penumbra";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};
struct RadianceTarget {
	static constexpr const char NAME[] = "Radiance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
	static constexpr bool REQUIRED = true;
};
struct InstanceImportanceSumTarget {
	static constexpr const char NAME[] = "Instance imp. Sum";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using CombinedTargets = TargetList<RadianceTarget, ImportanceTarget,
	InstanceImportanceSumTarget, PenumbraTarget>;

}}}} // namespace mufflon::renderer::decimaters::combined