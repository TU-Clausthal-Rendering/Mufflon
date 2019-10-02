#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer {

struct PScale {
	float m_curvScale = 1.0f;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Curvature Scale", ParameterTypes::FLOAT };
	}
};

struct PHeuristic {
	PARAM_ENUM(heuristic = Values::VCM, VCM, VCMPlus, VCMStar, IVCM);
	static constexpr ParamDesc get_desc() noexcept {
		return { "Heuristic", ParameterTypes::ENUM };
	}
};

struct FootprintTarget {
	static constexpr const char NAME[] = "Footprint";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using IvcmParameters = ParameterHandler<
	PSeed,
	PMinPathLength,
	PMaxPathLength,
	PMergeRadius,
	PProgressive,
	PHeuristic,
	PScale
>;

using IvcmTargets = TargetList<
	RadianceTarget,
	DensityTarget, FootprintTarget
>;

}} // namespace mufflon::renderer