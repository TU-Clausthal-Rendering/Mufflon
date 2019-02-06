#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PImportanceIterations {
	int iterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PTargetReduction {
	float reduction{ 0.875f };
	static ParamDesc get_desc() noexcept {
		return { "Target reduction", ParameterTypes::FLOAT };
	}
};

struct PShowSilhouette {
	bool showSilhouette{ false };
	static ParamDesc get_desc() noexcept {
		return { "Render shadow silhouette", ParameterTypes::BOOL };
	}
};

struct PVertexThreshold {
	int threshold{ 100 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation threshold", ParameterTypes::INT };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PTargetReduction, PVertexThreshold, PShowSilhouette, PMaxPathLength>;

}} // namespace mufflon::renderer