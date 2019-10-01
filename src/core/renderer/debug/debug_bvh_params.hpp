#pragma once

#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_targets.hpp"

namespace mufflon::renderer {

	struct PDebugBoxes {
		inline static constexpr const char* name = "AABBs";
		bool showBbox = false;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::BOOL };
		}
	};

	struct PDebugTopLevel {
		inline static constexpr const char* name = "TLAS";
		bool showTopLevel = false;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::BOOL };
		}
	};

	struct PDebugBotLevel {
		inline static constexpr const char* name = "BLAS instance";
		int botLevelIndex = -1;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::INT };
		}
	};

	struct PDebugLevelHighlight {
		inline static constexpr const char* name = "Level highlight";
		int levelHighlight = -1;
		static constexpr  ParamDesc get_desc() noexcept {
			return  { name, ParameterTypes::INT };
		}
	};

	struct PDebugColorInstance {
		inline static constexpr const char* name = "Instance color coding";
		bool colorInstance = false;
		static constexpr ParamDesc get_desc() noexcept {
			return { name, ParameterTypes::BOOL };
		}
	};

	using DebugBvhParameters = ParameterHandler<PDebugBoxes, PDebugTopLevel, PDebugBotLevel, PDebugLevelHighlight, PDebugColorInstance>;

	using DebugBvhTargets = TargetList<RadianceTarget>;

} // namespace mufflon::render#pragma once
