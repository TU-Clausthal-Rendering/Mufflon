#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using LtParameters = ParameterHandler<
	PMinPathLength,
	PMaxPathLength
>;

}} // namespace mufflon::renderer