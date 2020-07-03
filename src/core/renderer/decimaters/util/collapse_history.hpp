#pragma once

#include "util/int_types.hpp"

namespace mufflon::renderer::decimaters {

struct CollapseHistory {
	u32 collapsedTo;
	u32 frameIndex;
	bool collapsed;
};

} // namespace mufflon::renderer::decimaters