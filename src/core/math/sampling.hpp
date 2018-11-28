#pragma once

#include "ei/3dtypes.hpp"
#include "export/api.hpp"
#include "core/scene/types.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace math {

struct DirectionSample {
	scene::Direction direction;
	AngularPdf pdf;
};

struct PositionSample {
	scene::Point position;
	AreaPdf pdf;
};

// Evaluate PDF
CUDA_FUNCTION constexpr AngularPdf get_uniform_dir_pdf() {
	return AngularPdf{ 1.f / (4.f * ei::PI) };
}
CUDA_FUNCTION constexpr AngularPdf get_uniform_cone_pdf(float cosThetaMax) {
	return AngularPdf{ 1.f / (2.f * ei::PI * (1.f - cosThetaMax)) };
}
CUDA_FUNCTION constexpr AngularPdf get_cosine_dir_pdf(float cosTheta) {
	return AngularPdf{ cosTheta / ei::PI };
}

// Basic transformation of an uniform integer into an uniform flaot in [0,1)
CUDA_FUNCTION __forceinline__ float sample_uniform(u32 u0) {
	return u0 / 4294967810.0f; // successor(float(0xffffffff));
}
CUDA_FUNCTION __forceinline__ ei::Vec2 sample_uniform(u64 u01) {
	return ei::Vec2{ sample_uniform(u32(u01 & 0xffffffffull)),
					 sample_uniform(u32(u01 >> 32ull)) };
}

// Rescale a random number inside the interval [pLeft, pRight] to [0,1]
CUDA_FUNCTION __forceinline__ float rescale_sample(float x, float pLeft, float pRight) {
	return (x - pLeft) / (pRight - pLeft);
}

// Sample a cosine distributed direction from two random numbers in [0,1)
CUDA_FUNCTION __forceinline__ DirectionSample sample_dir_cosine(float u0, float u1) {
	float cosTheta = sqrt(u0);			// cos(acos(sqrt(x))) = sqrt(x)
	float sinTheta = sqrt(1.0f - u0);	// sqrt(1-cos(theta)^2)
	float phi = u1 * 2 * ei::PI;
	return DirectionSample{ scene::Direction { sinTheta * sin(phi), sinTheta * cos(phi), cosTheta },
							get_cosine_dir_pdf(cosTheta) };
}

// Sample a uniformly distributed direction (full sphere) from two random numbers in [0,1)
CUDA_FUNCTION __forceinline__ DirectionSample sample_dir_sphere_uniform(float u0, float u1) {
	// u0 should include 1
	float cosTheta = u0 * 2.0f - 1.0f;
	float sinTheta = sqrt((1.0f - cosTheta) * (1.0f + cosTheta));
	float phi = u1 * 2.0f * ei::PI;
	return DirectionSample{ scene::Direction{ sinTheta * sin(phi),
											  sinTheta * cos(phi), cosTheta },
							get_uniform_dir_pdf() };
}

// Sample a cone uniformly; TODO: importance sampling?
CUDA_FUNCTION __forceinline__ DirectionSample sample_cone_uniform(float maxCosTheta,
														  float u0, float u1) {
	float cosTheta = (1.f - u0) + u0 * maxCosTheta;
	float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
	float phi = 2.f * u1 * ei::PI;
	return DirectionSample{ scene::Direction{ cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta },
							get_uniform_cone_pdf(maxCosTheta) };
}

// Samples a position on a triangle
// The third barycentric is ommitted. It is simply 1-s-t.
CUDA_FUNCTION __forceinline__ ei::Vec2 sample_barycentric(float u0, float u1) {
	// TODO: test the sqrt-free variant
	float s = 1.f - sqrtf(1.f - u0);
	float t = (1.f - s) * u1;
	return ei::Vec2{s, t};
}

// Samples a position within the bounding box orthogonally projected in the given direction
// Taken from Johannes' renderer
CUDA_FUNCTION __forceinline__ PositionSample sample_position(const scene::Direction& dir,
															 const ei::Box& bounds,
															 float u0, float u1) {
	// Compute projected cube area
	ei::Vec3 sides = bounds.max - bounds.min;
	float projAx = sides.y * sides.z * fabsf(dir.x);
	float projAy = sides.x * sides.z * fabsf(dir.y);
	float projAz = sides.x * sides.y * fabsf(dir.z);
	float area = projAx + projAy + projAz;
	// Sample a position on one of the cube faces
	ei::Vec3 position;
	float x = u0 * area;
	if(x < projAx) {
		u0 = rescale_sample(u0, 0.0f, projAx);
		position = ei::Vec3{ (dir.x < 0.f) ? 1.f : 0.f, u1, u0 };
	} else if(x < projAx + projAy) {
		u0 = rescale_sample(u0, projAx, projAx + projAy);
		position = ei::Vec3{ u1, (dir.y < 0.f) ? 1.f : 0.f, u0 };
	} else {
		u0 = rescale_sample(u0, projAx + projAy, 1.0f);
		position = ei::Vec3{ u1, u0, (dir.z < 0.f) ? 1.f : 0.f };
	}
	return PositionSample{ bounds.min + position * sides, AreaPdf{ 1.f / area } };
}

/*
 * A RndSet is a fixed size set of random numbers which may be consumed by a sampler.
 * There are different sets dependent on the application. Most samplers use 2 numbers.
 * For samplers with layer decisions (light tree, materials) there is a rndset with
 * an additional 64bit random value (RndSet2_1).
 */
struct RndSet2 {
	float u0;
	float u1;

	RndSet2(u64 x) :
		u0{sample_uniform(u32(x))},
		u1{sample_uniform(u32(x >> 32))}
	{}
};

struct RndSet2_1 {
	float u0;
	float u1;
	u64 i0;

	RndSet2_1(u64 x0, u64 x1) :
		u0{sample_uniform(u32(x0))},
		u1{sample_uniform(u32(x0 >> 32))},
		i0{x1}
	{}
};

}} // namespace mufflon::math