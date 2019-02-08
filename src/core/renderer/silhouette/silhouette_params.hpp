#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PImportanceIterations {
	int iterations{ 1 };
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

struct PDecimationEnabled {
	bool decimationEnabled{ true };
	static ParamDesc get_desc() noexcept {
		return { "Enable decimation", ParameterTypes::BOOL };
	}
};

struct PEnableDirectImportance {
	bool enableDirectImportance{ true };
	static ParamDesc get_desc() noexcept {
		return { "Enable direct importance", ParameterTypes::BOOL};
	}
};

struct PEnableSilhouetteImportance {
	bool enableSilhouetteImportance{ true };
	static ParamDesc get_desc() noexcept {
		return { "Enable silhouette importance", ParameterTypes::BOOL };
	}
};

struct PMaxNormalDeviation {
	float maxNormalDeviation{ 60.f };
	static ParamDesc get_desc() noexcept {
		return { "Maximum normal deviation", ParameterTypes::FLOAT};
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PTargetReduction, PVertexThreshold,
	PMaxNormalDeviation, PEnableDirectImportance, PEnableSilhouetteImportance, PDecimationEnabled,
	PShowSilhouette, PMaxPathLength>;

}} // namespace mufflon::renderer