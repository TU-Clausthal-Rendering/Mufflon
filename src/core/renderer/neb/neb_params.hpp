#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PNeeMergeRadius {
	float neeMergeRadius { 0.001f };
	static ParamDesc get_desc() noexcept {
		return {"NEE merge radius", ParameterTypes::FLOAT};
	}
};

using NebParameters = ParameterHandler<
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide,
	PMergeRadius,
	PNeeMergeRadius
>;

}} // namespace mufflon::renderer