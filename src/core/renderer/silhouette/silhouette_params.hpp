#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer {

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

struct PMemoryConstraint {
	int memoryConstraint = 100'000;
	static ParamDesc get_desc() noexcept {
		return { "Memory available for geometry", ParameterTypes::INT };
	}
};

struct PInitialConstraint {
	bool isConstraintInitial = false;
	static ParamDesc get_desc() noexcept {
		return { "Is memory constraint for initial decimation", ParameterTypes::BOOL };
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

using SilhouetteParameters = ParameterHandler<PImportanceIterations, PDecimationIterations, PTargetReduction, PVertexThreshold,
	PDirectIndirectRatio, PDecimationEnabled, PMemoryConstraint, PInitialConstraint, PMaxPathLength,
	PDirectImportance, PIndirectImportance, PEyeImportance>;

}} // namespace mufflon::renderer