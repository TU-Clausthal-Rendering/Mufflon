#pragma once

#include "util/log.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include <cuda_runtime.h>


namespace mufflon { namespace cameras {

/*
 * Each camera must implement a sampler which is called by this function based on the type.
 * coord: integer coordinate (0-indexed) of the pixel.
 * resolution: total framebuffer resolution (integral, but converted to float)
 * rndSet: a set with 4 random numbers for the camera sampling.
 */
__host__ __device__ RaySample
sample_ray(const CameraParams& params, const Pixel& coord, const ei::Vec2& resolution, const RndSet& rndSet) {
	switch(params.type) {
		case CameraModel::PINHOLE: {
			sample_ray(static_cast<const PinholeParams&>(params), coord, resolution, rndSet)
		} break;
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[cameras::sample_ray] Trying to evaluate unimplemented camera model ", params.type);
#endif
	}
}

// Compute pixel position and PDF
// position: a direction in world space.
__host__ __device__ ProjectionResult
project(const PinholeParams& params, const ei::Vec2& resolution, const scene::Point& position) {
	switch(params.type) {
		case CameraModel::PINHOLE: {
			project(static_cast<const PinholeParams&>(params), resolution, position)
		} break;
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[cameras::project] Trying to evaluate unimplemented camera model ", params.type);
#endif
	}
}

}} // namespace mufflon::cameras