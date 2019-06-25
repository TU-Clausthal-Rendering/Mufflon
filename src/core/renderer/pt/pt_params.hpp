#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PSplitFactor {
	float split = 0.5f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Vertical CPU/CUDA split", ParameterTypes::FLOAT };
	}
};

using PtParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide
>;

using HybridPtParams = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide,
	PSplitFactor
>;

}} // namespace mufflon::renderer