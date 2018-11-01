#pragma once

#include "core/scene/types.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace math {

// Sample a cosine distributed direction from two random numbers in [0,1)
__host__ __device__ scene::Direction sample_dir_cosine(float u0, float u1) {
	float cosTheta = sqrt(u0);			// cos(acos(sqrt(x))) = sqrt(x)
	float sinTheta = sqrt(1.0f - u0);	// sqrt(1-cos(theta)^2)
	float phi = u1 * 2 * ei::PI;
	return scene::Direction { sinTheta * sin(phi), sinTheta * cos(phi), cosTheta };
}

}} // namespace mufflon::math