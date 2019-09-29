#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon::renderer {


	struct PDebugTopLevel {
		inline static constexpr const char* name = "Top level";
		bool showTopLevel = false;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::BOOL };
		}
	};

	struct PDebugBotLevel {
		inline static constexpr const char* name = "Bot level instance";
		int botLevelIndex = -1;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::INT };
		}
	};

	using DebugBvhParameters = ParameterHandler<PDebugTopLevel, PDebugBotLevel>;

	using DebugBvhTargets = TargetList<RadianceTarget>;

} // namespace mufflon::render#pragma once
