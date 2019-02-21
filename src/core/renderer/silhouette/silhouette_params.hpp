#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

struct PDecimationIterations {
	int decimationIterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PEnableViewImportance {
	bool enableViewImportance{ true };
	static ParamDesc get_desc() noexcept {
		return { "Enable view importance", ParameterTypes::BOOL };
	}
};

struct PEnableIndirectImportance {
	bool enableIndirectImportance{ true };
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

struct PUseRadianceWeightedImportance {
	bool useRadianceWeightedImportance{ true };
	static ParamDesc get_desc() noexcept {
		return { "Use radiance-weighted importance", ParameterTypes::BOOL };
	}
};

struct PDirectIndirectRatio {
	float directIndirectRatio{ 0.02f };
	static ParamDesc get_desc() noexcept {
		return { "Ratio threshold for direct/indirect illumination", ParameterTypes::FLOAT };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PVertexThreshold,
	PEnableViewImportance, PEnableIndirectImportance, PEnableSilhouetteImportance, PUseRadianceWeightedImportance,
	PDirectIndirectRatio, PDecimationEnabled, PKeepImportance, PMaxPathLength>;

}} // namespace mufflon::renderer