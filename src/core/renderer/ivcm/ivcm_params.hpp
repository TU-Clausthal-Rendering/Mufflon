#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using IvcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive
>;

}} // namespace mufflon::renderer