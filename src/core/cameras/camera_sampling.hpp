#pragma once

#include "util/log.hpp"
#include "export/api.hpp"
#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include <cuda_runtime.h>


namespace mufflon { namespace cameras {

/*
 * Sample a position on the lens or near plane from where a camera ray may start.
 * The second sampler camera_sample_ray will receive this output as intput (exitPosWorld)
 * to complete the ray sample.
 */
CUDA_FUNCTION math::PositionSample
camera_sample_position(const CameraParams& params, const Pixel& pixel, const math::RndSet2& rndSet) {
	switch(params.type) {
		case CameraModel::PINHOLE: {
			return pinholecam_sample_position(static_cast<const PinholeParams&>(params), pixel, rndSet);
		}
		default:
#ifndef __CUDA_ARCH__
			logWarning("[cameras::camera_sample_position] Trying to evaluate unimplemented camera model ", params.type);
#endif
			return math::PositionSample{};
	}
}

/*
 * Sample an outgoing direction given an exit point on the lens/near plane, which
 * was produced by camera_sample_position.
 * exitPosWorld: Generated point from camera_sample_position.
 * rndSet: a set with 4 random numbers for the camera sampling.
 */
CUDA_FUNCTION Importon
camera_sample_ray(const CameraParams& params, const scene::Point& exitPosWorld, const math::RndSet2& rndSet) {
	switch(params.type) {
		case CameraModel::PINHOLE: {
			return pinholecam_sample_ray(static_cast<const PinholeParams&>(params), exitPosWorld);
		}
		default:
#ifndef __CUDA_ARCH__
			logWarning("[cameras::sample_ray] Trying to evaluate unimplemented camera model ", params.type);
#endif
			return Importon{};
	}
}

// Compute pixel position and PDF
// excident: a direction in world space (from camera to object).
CUDA_FUNCTION ProjectionResult
camera_project(const CameraParams& params, const scene::Point& excident) {
	switch(params.type) {
		case CameraModel::PINHOLE: {
			return pinholecam_project(static_cast<const PinholeParams&>(params), excident);
		}
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[cameras::project] Trying to evaluate unimplemented camera model ", params.type);
#endif
			return ProjectionResult{};
	}
}

constexpr std::size_t MAX_CAMERA_PARAM_SIZE = sizeof(PinholeParams); // TODO max() with the packs of the other camera models

}} // namespace mufflon::cameras