#pragma once

#include "core/renderer/parameter.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette {

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

struct PSharpnessFactor {
	float sharpnessFactor = 10.f;
	static ParamDesc get_desc() noexcept {
		return { "2/(1+exp(-BxDF/factor)) - 1", ParameterTypes::FLOAT };
	}
};

struct PViewWeight {
	float viewWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of view paths", ParameterTypes::FLOAT };
	}
};

struct PLightWeight {
	float lightWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of light paths", ParameterTypes::FLOAT };
	}
};

struct PShadowWeight {
	float shadowWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow paths", ParameterTypes::FLOAT };
	}
};

struct PShadowSilhouetteWeight {
	float shadowSilhouetteWeight = 1.f;
	static ParamDesc get_desc() noexcept {
		return { "Imp. weight of shadow silhouette paths", ParameterTypes::FLOAT };
	}
};

struct PRenderUpdate {
	bool renderUpdate = false;
	static ParamDesc get_desc() noexcept {
		return { "Show update between decimations", ParameterTypes::BOOL };
	}
};

using SilhouetteParameters = ParameterHandler<
	PImportanceIterations, PDecimationIterations,
	PTargetReduction, PInitialReduction, PVertexThreshold,
	PDirectIndirectRatio, PSharpnessFactor,
	PViewWeight, PLightWeight, PShadowWeight, PShadowSilhouetteWeight,
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide,
	PRenderUpdate
>;

}}}} // namespace mufflon::renderer::decimaters::silhouette