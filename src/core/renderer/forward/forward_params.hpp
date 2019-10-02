#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon::renderer {


struct PForwardTopLevel {
	inline static constexpr const char* name = "Top level";
	bool showTopLevel = false;
	static constexpr ParamDesc get_desc() noexcept {
		return { name, ParameterTypes::BOOL };
	}
};

struct PForwardBotLevel {
	inline static constexpr const char* name = "Bot level instance";
	int botLevelIndex = -1;
	static constexpr ParamDesc get_desc() noexcept {
		return { name, ParameterTypes::INT };
	}
};
	
using ForwardParameters = ParameterHandler<PForwardTopLevel, PForwardBotLevel>;

using ForwardTargets = TargetList<RadianceTarget, PositionTarget, NormalTarget>;

} // namespace mufflon::renderer