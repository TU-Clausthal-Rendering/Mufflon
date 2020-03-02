#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"
#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace combined {


struct PClusterSize {
	int gridRes{ 0 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Cluster grid resolution", ParameterTypes::INT };
	}
};
struct PImpStructCapacity {
	int impCapacity{ 1024 * 1024 * 4 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance data capacity", ParameterTypes::INT };
	}
};
struct PImpDataStruct {
	PARAM_ENUM(impDataStruct, VERTEX, HASHGRID, OCTREE) = Values::OCTREE;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Importance data structure", ParameterTypes::ENUM };
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
	silhouette::PImportanceIterations, silhouette::PTargetReduction,
	silhouette::PInitialReduction, silhouette::PVertexThreshold,
	silhouette::PSelectiveImportance, PImpSumStrat,
	PClusterSize, PImpStructCapacity,
	PVertexDistMethod, PImpWeightMethod,
	silhouette::PViewWeight, silhouette::PLightWeight,
	silhouette::PShadowWeight, silhouette::PShadowSilhouetteWeight,
	PSlidingWindow,
	PMaxPathLength, PNeeCount
>;

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
struct ImportanceSumTarget {
	static constexpr const char NAME[] = "Importance Sum";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using CombinedTargets = TargetList<RadianceTarget, silhouette::ImportanceTarget, ImportanceSumTarget, PenumbraTarget>;

}}}} // namespace mufflon::renderer::decimaters::combined