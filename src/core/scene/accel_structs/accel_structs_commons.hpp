#pragma once

#include "util/types.hpp"
#include "util/assert.hpp"
#include "export/api.hpp"

namespace mufflon {
namespace scene {
namespace accel_struct {

CUDA_FUNCTION constexpr void extract_prim_counts(i32 primitiveCount, ei::IVec4& count) {
	i32 sphCountMask = 0x000003FF;
	i32 triCountMask = 0x3FF00000;
	i32 quadCountMask = 0x000FFC00;
	i32 triShift = 20;
	i32 quadShift = 10;
	count.x = (primitiveCount & triCountMask) >> triShift;
	count.y = (primitiveCount & quadCountMask) >> quadShift;
	count.z = (primitiveCount & sphCountMask);
	count.w = count.x + count.y + count.z;
}

}
}
}