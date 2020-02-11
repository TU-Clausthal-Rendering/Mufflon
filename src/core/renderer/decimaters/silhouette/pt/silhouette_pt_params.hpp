#pragma once

#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

struct PClusterSize {
	int gridRes{ 0 };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Cluster grid resolution", ParameterTypes::INT };
	}
};
struct PImpStructCapacity {
	int impCapacity{ 1024*1024*4 };
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

using SilhouetteParameters = ParameterHandler <
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PSelectiveImportance, PShadowSizeWeight, PImpDataStruct,
	PDirectIndirectRatio, PSharpnessFactor,
	PClusterSize, PImpStructCapacity,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide
>;

struct ShadowOmitted {
	static constexpr const char NAME[] = "Shadow Omitted";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

struct ShadowRecorded {
	static constexpr const char NAME[] = "Shadow Recorded";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};


using SilhouetteTargets = TargetList<ImportanceTarget, PolyShareTarget, ShadowRecorded, ShadowOmitted>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt