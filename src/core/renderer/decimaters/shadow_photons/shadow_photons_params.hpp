#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace spm {

struct PInterpolate {
	bool pointSampling = true;
	static constexpr ParamDesc get_desc() {
		return ParamDesc{ "Point sampling", ParameterTypes::BOOL };
	}
};

using ShadowPhotonParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PInterpolate
>;

}}}} // namespace mufflon::renderer::decimaters::spm