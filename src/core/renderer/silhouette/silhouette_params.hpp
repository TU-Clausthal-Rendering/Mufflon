#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PImportanceIterations {
	int importanceIterations{ 1 };
	static ParamDesc get_desc() noexcept {
		return { "Importance iterations", ParameterTypes::INT };
	}
};

struct PDecimationIterations {
	int decimationIterations{ 1 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PTargetReduction {
	float reduction{ 0.875f };
	static ParamDesc get_desc() noexcept {
		return { "Target reduction", ParameterTypes::FLOAT };
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

struct PEnableIndirectImportance {
	bool enableIndirectImportance{ false };
	static ParamDesc get_desc() noexcept {
		return { "Enable indirect importance", ParameterTypes::BOOL};
	}
};

struct PEnableSilhouetteImportance {
	bool enableSilhouetteImportance{ true };
	static ParamDesc get_desc() noexcept {
		return { "Enable silhouette importance", ParameterTypes::BOOL };
	}
};

struct PKeepImportance {
	bool keepImportance{ false };
	static ParamDesc get_desc() noexcept {
		return { "Keep importance across resets", ParameterTypes::BOOL };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PVertexThreshold,
	PEnableIndirectImportance, PEnableSilhouetteImportance, PDecimationEnabled, PKeepImportance, PMaxPathLength>;

}} // namespace mufflon::renderer