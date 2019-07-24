#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using PtParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PNeeCount,
	PNeePositionGuide
>;

}} // namespace mufflon::renderer