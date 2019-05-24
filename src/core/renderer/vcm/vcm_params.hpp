#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

using VcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive
>;

}} // namespace mufflon::renderer