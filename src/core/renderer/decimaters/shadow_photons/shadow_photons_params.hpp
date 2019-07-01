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
	PARAM_ENUM(mode, HASHGRID, OCTREE, HEURISTIC) = Values::HASHGRID;
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

using ShadowPhotonParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PInterpolate,
	PSpvMode,
	PCellSize,
	PSplitFactor
>;

}}}} // namespace mufflon::renderer::decimaters::spm