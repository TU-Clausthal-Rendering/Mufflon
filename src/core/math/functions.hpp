#pragma once
#include "core/export/api.h"
#include <ei/elementarytypes.hpp>

namespace mufflon { namespace math {

// inverse erf approximation precision: +-6e-3
CUDA_FUNCTION float erfInv(float x) {
	float tt1, tt2, lnx, sgn;
	sgn = (x < 0.f) ? -1.0f : 1.0f;
	lnx = logf((1.0f - x) * (1.0f + x));
	tt1 = 2.f / (ei::PI*0.147f) + 0.5f * lnx;
	tt2 = 1.f / (0.147f) * lnx;
	return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}

}} // namespace mufflon::math