#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PkNN {
	int knn { 0 };
	static ParamDesc get_desc() noexcept {
		return {"kNN Merges", ParameterTypes::INT};
	}
};

using BpmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive,
	PkNN
>;

}} // namespace mufflon::renderer