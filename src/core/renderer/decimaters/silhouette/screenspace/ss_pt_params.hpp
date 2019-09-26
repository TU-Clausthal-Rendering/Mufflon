#pragma once

#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

struct PPenumbraWeight {
	PARAM_ENUM(penumbraWeight, LDIVDP3, SMEAR, SMEARDIV2, SMEARDIV3) = Values::LDIVDP3;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Penumbra weight", ParameterTypes::ENUM };
	}
};

using SilhouetteParameters = ParameterHandler<
	PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide
>;

struct RadianceTarget {
	static constexpr const char NAME[] = "Radiance";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 3u;
	static constexpr bool REQUIRED = true;
};
struct ShadowTarget {
	static constexpr const char NAME[] = "Shadow Areas";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};
struct PenumbraTarget {
	static constexpr const char NAME[] = "Shadow Areas";
	using PixelType = float;
	static constexpr u32 NUM_CHANNELS = 1u;
};

using SilhouetteTargets = TargetList<
	RadianceTarget, ShadowTarget
>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss