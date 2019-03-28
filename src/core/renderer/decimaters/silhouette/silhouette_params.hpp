#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon::renderer::decimaters::silhouette {

struct PDecimationIterations {
	int decimationIterations{ 10 };
	static ParamDesc get_desc() noexcept {
		return { "Decimation iterations", ParameterTypes::INT };
	}
};

struct PDirectIndirectRatio {
	float directIndirectRatio{ 0.02f };
	static ParamDesc get_desc() noexcept {
		return { "Ratio threshold for direct/indirect illumination", ParameterTypes::FLOAT };
	}
};

struct PDirectImportance {
	bool enableDirectImportance = true;
	static ParamDesc get_desc() noexcept {
		return { "Enable direct importance", ParameterTypes::BOOL };
	}
};

struct PIndirectImportance {
	bool enableIndirectImportance = true;
	static ParamDesc get_desc() noexcept {
		return { "Enable indirect importance", ParameterTypes::BOOL };
	}
};

struct PEyeImportance {
	bool enableEyeImportance = true;
	static ParamDesc get_desc() noexcept {
		return { "Enable eye importance", ParameterTypes::BOOL };
	}
};

struct PNormalDeviation {
	float maxNormalDeviation = 60.f;
	static ParamDesc get_desc() noexcept {
		return { "Max. normal deviation from collapse", ParameterTypes::FLOAT};
	}
};

struct PCollapseMode {
	int collapseMode = 0;
	static ParamDesc get_desc() noexcept {
		return { "Collapse mode (0 = none, 1 = no concave, 2 = remember silhouettes, 3 = dampen importance)", ParameterTypes::INT };
	}
};

struct PDisplayProjection {
	bool displayProjection = true;
	static ParamDesc get_desc() noexcept {
		return { "Displays the projected importance/silhouettes instead", ParameterTypes::BOOL };
	}
};

struct PResetOnReload {
	bool resetOnReload = true;
	static ParamDesc get_desc() noexcept {
		return { "Reset the mesh/importance on reload", ParameterTypes::BOOL };
	}
};

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PInitialReduction,
	PVertexThreshold, PDirectIndirectRatio, PNormalDeviation, PCollapseMode, PMaxPathLength,
	PDirectImportance, PIndirectImportance, PEyeImportance, PDisplayProjection, PResetOnReload>;

} // namespace mufflon::renderer::decimaters::silhouette