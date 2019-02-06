#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PImportanceIterations {
	int iterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PShowSilhouette {
	bool showSilhouette{ false };
	static ParamDesc get_desc() noexcept {
		return { "Render shadow silhouette", ParameterTypes::BOOL };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PShowSilhouette, PMaxPathLength>;

}} // namespace mufflon::renderer