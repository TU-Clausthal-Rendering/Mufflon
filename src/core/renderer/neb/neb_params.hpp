#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using NebParameters = ParameterHandler<
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide,
	PMergeRadius
>;

}} // namespace mufflon::renderer