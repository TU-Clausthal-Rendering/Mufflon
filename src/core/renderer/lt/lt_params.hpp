#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using LtParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength
>;

}} // namespace mufflon::renderer