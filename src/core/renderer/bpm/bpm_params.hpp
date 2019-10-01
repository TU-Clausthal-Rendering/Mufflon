#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

struct PkNN {
	int knn { 0 };
	static constexpr ParamDesc get_desc() noexcept {
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

using BpmTargets = TargetList<
	RadianceTarget
>;

}} // namespace mufflon::renderer