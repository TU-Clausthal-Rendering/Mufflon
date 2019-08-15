#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_target.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace spm {

struct PInterpolate {
	PARAM_ENUM(interpolation, POINT, LINEAR, SMOOTHSTEP) = Values::LINEAR;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Density interpolation", ParameterTypes::ENUM };
	}
};

struct PSpvMode {
	PARAM_ENUM(mode, HASHGRID, OCTREE) = Values::HASHGRID;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Shadow mode", ParameterTypes::ENUM };
	}
};

struct PCellSize {
	float cellSize = 0.05f;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Hashgrid cell size", ParameterTypes::FLOAT };
	}
};

struct PSplitFactor {
	float splitFactor = 2.f;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Octree cell split factor", ParameterTypes::FLOAT };
	}
};

struct PBalanceOctree {
	bool balanceOctree = false;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Balance octree every iteration", ParameterTypes::BOOL };
	}
};

using ShadowPhotonParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PInterpolate,
	PSpvMode,
	PCellSize,
	PSplitFactor,
	PBalanceOctree
>;


struct ImportanceTarget {
	static constexpr const char NAME[] = "Importance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct LightDensityTarget {
	static constexpr const char NAME[] = "Photon density";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};
struct ShadowDensityTarget {
	static constexpr const char NAME[] = "Shadow photon density";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};
struct ShadowGradientTarget {
	static constexpr const char NAME[] = "Shadow photon gradient";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
};

using ShadowPhotonTargets = TargetList<ImportanceTarget, LightDensityTarget, ShadowDensityTarget, ShadowGradientTarget>;

}}}} // namespace mufflon::renderer::decimaters::spm