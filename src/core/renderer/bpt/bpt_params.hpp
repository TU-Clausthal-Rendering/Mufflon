#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using BptParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength
>;

}} // namespace mufflon::renderer