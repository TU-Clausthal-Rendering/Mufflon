#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace spm {

struct PInterpolate {
	bool pointSampling = true;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Point sampling", ParameterTypes::BOOL };
	}
};

struct PCellSize {
	float cellSize = 1.f;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Hashgrid cell size", ParameterTypes::FLOAT };
	}
};

using ShadowPhotonParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PInterpolate,
	PCellSize
>;

}}}} // namespace mufflon::renderer::decimaters::spm