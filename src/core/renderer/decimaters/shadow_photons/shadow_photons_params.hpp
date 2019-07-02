#pragma once

#include "core/renderer/parameter.hpp"

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

struct PUseHeuristic {
	bool useHeuristic = false;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Use heuristic for light size instead of gradient", ParameterTypes::BOOL };
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
	PUseHeuristic,
	PCellSize,
	PSplitFactor,
	PBalanceOctree
>;

}}}} // namespace mufflon::renderer::decimaters::spm