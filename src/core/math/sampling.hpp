#pragma once

#include "core/scene/types.hpp"
#include "ei/3dtypes.hpp"
#include <cuda_runtime.h>
#include "export/api.hpp"

namespace mufflon { namespace math {

struct DirectionSample {
	scene::Direction direction;
	AngularPdf pdf;
};

struct PositionSample {
	scene::Point position;
	AreaPdf pdf;
};

// Sample a cosine distributed direction from two random numbers in [0,1)
CUDA_FUNCTION __forceinline__ DirectionSample sample_dir_cosine(float u0, float u1) {
	float cosTheta = sqrt(u0);			// cos(acos(sqrt(x))) = sqrt(x)
	float sinTheta = sqrt(1.0f - u0);	// sqrt(1-cos(theta)^2)
	float phi = u1 * 2 * ei::PI;
	return DirectionSample{ scene::Direction { sinTheta * sin(phi), sinTheta * cos(phi), cosTheta },
					AngularPdf{1.f / (2.f * ei::PI * ei::PI)} };
}

// Sample a uniformly distributed direction (full sphere) from two random numbers in [0,1)
CUDA_FUNCTION __forceinline__ DirectionSample sample_dir_sphere_uniform(float u0, float u1) {
	// u0 should include 1
	float cosTheta = u0 * 2.0f - 1.0f;
	float sinTheta = sqrt((1.0f - cosTheta) * (1.0f + cosTheta));
	float phi = u1 * 2.0f * ei::PI;
	return DirectionSample{ scene::Direction{ sinTheta * sin(phi),
											  sinTheta * cos(phi), cosTheta },
							AngularPdf{1.f / (4.f * ei::PI)} };
}

CUDA_FUNCTION __forceinline__ DirectionSample sample_cone(float maxCosTheta,
														  float u0, float u1) {
	float cosTheta = (1.f - u0) + u0 * maxCosTheta;
	float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
	float phi = 2.f * u1 * ei::PI;
	return DirectionSample{ scene::Direction{ cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta },
							AngularPdf{1.f / (2.f * ei::PI * (1.f - maxCosTheta))} };
}

// Samples a position on a triangle
CUDA_FUNCTION __forceinline__ PositionSample sample_position(const ei::Triangle& tri,
															 float u0, float u1) {
	float s = 1.f - sqrtf(1.f - u0);
	float t = (1.f - s) * u1;
	// From PBRT
	return PositionSample{ scene::Point{tri.v0 + s * (tri.v1 - tri.v0) + t * (tri.v2 - tri.v0) },
						   AreaPdf{1.f / ei::surface(tri)} };
}

// Samples a position within the bounding box orthogonally projected in the given direction
// Taken from Johannes' renderer
CUDA_FUNCTION __forceinline__ PositionSample sample_position(const scene::Direction& dir,
															 const ei::Box& bounds,
															 float u0, float u1, float u2) {
	// Compute projected cube area
	ei::Vec3 sides = bounds.max - bounds.min;
	float projAx = sides.y * sides.z * fabsf(dir.x);
	float projAy = sides.x * sides.z * fabsf(dir.y);
	float projAz = sides.x * sides.y * fabsf(dir.z);
	float area = projAx + projAy + projAz;
	// Sample a position on one of the cube faces
	ei::Vec3 position;
	float x = u0 * area;
	if(x < projAx)
		position = ei::Vec3{ (dir.x < 0.f) ? 1.f : 0.f, u1, u2 };
	else if(x < projAx + projAy)
		position = ei::Vec3{ u1, (dir.y < 0.f) ? 1.f : 0.f, u2 };
	else
		position = ei::Vec3{ u1, u2, (dir.z < 0.f) ? 1.f : 0.f };
	return PositionSample{ bounds.min + position * sides, AreaPdf(1.f / area) };
}

// TODO: evaluate PDF


}} // namespace mufflon::math