#pragma once

#include "ei/3dtypes.hpp"
#include "core/export/api.h"
#include "core/scene/types.hpp"
#include "sample_types.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace math {

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

// Computes u64 * [0,1] in fixed point.
// Semantic like: u64(num * p), however doing this directly leads to invalid
// conversions (num >= 2^63 -> 2^63).
CUDA_FUNCTION __forceinline__ u64 percentage_of(u64 num, float p) {
	// Get a fixed point 31 number.
	// Multiplying with 2^32 would cause an overflow for p=1.
	u64 pfix = u64(p * 2147483648.0f);
	// Multiply low and high part independently and shift the results.
	return ((pfix * (num & 0x7fffffff)) >> 31)	// Low 31 bits
		  + (pfix * (num >> 31));				// High 33 bits
}

/*CUDA_FUNCTION __forceinline__ u64 mul32_3232(u32 a, u32 b0, u32 b1) {
	return u64(a) * b0 + ((u64(a) * b1) >> 32ull);
}
// BUGGY!
// Integer rounding of the second terms looses too much...
CUDA_FUNCTION __forceinline__ u64 div64_3232(u64 a, u32 b0, u32 b1) {
	return a / b0 - ((a * b1) >> 32ull) / mul32_3232(b0, b0, b1);
}
CUDA_FUNCTION u64 div128_64(u64 a0, u64 a1, u64 b) {
	u32 b0 = b >> 32ull;
	u32 b1 = b & 0xffffffffull;
	u64 hi = div64_3232(a0, b0, b1);
	mAssert(hi < 0x100000000);
	u64 r = a0 - mul32_3232(hi, b0, b1);
	mAssert(r < 0x100000000);
	u64 am = (r << 32ull) | (a1 >> 32ull);
	u32 lo = div64_3232(am, b0, b1);
	r = am - mul32_3232(lo, b0, b1);
	u64 al = (r << 32ull) | (a1 & 0xffffffffull);
	lo += div64_3232(al, b0, b1) >> 32ull;
	return hi << 32ull | lo;
}*/

// Rescale a random number inside the interval [pLeft, pRight] to [0,u64::max]
CUDA_FUNCTION __forceinline__ u64 rescale_sample(u64 x, u64 pLeft, u64 pRight) {
	u64 interval = pRight - pLeft;
	u64 xrel = x - pLeft;
	u64 uMax = std::numeric_limits<u64>::max();
	// Problem: (xrel * u64::max) / interval leaves the 64 bit range
	//			 xrel * (u64::max / interval) is imprecise (up to a factor of 2)
	// To correctly rescale we would need to compute (xrel * uMax) / interval which
	// requires a 128 multiplication and a 128/64 division. This seems far more expensive
	// than the double solution. Although, the double solution looses some bits in the
	// conversion.
	//return u64(double(xrel) / double(interval) * uMax);
	// Solution with a double which looses less bits
	return xrel * (uMax / interval) + u64(xrel * double(uMax % interval) / interval);
	//return xrel * (uMax / interval) + u64(xrel * float(uMax % interval) / interval);
	// For the log (at least commited once) here the other tried solutions
	//u64 res = div128_64(xrel, 0, interval);
	//return res;
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
	float projAx = sides.y * sides.z * ei::abs(dir.x);
	float projAy = sides.x * sides.z * ei::abs(dir.y);
	float projAz = sides.x * sides.y * ei::abs(dir.z);
	float area = projAx + projAy + projAz;
	// Sample a position on one of the cube faces
	ei::Vec3 position;
	float x = u0 * area;
	if(x < projAx) {
		x = rescale_sample(x, 0.0f, projAx);
		position = ei::Vec3{ (dir.x < 0.f) ? 1.f : 0.f, u1, x };
	} else if(x < projAx + projAy) {
		x = rescale_sample(x, projAx, projAx + projAy);
		position = ei::Vec3{ u1, (dir.y < 0.f) ? 1.f : 0.f, x };
	} else {
		x = rescale_sample(x, projAx + projAy, area);
		position = ei::Vec3{ u1, x, (dir.z < 0.f) ? 1.f : 0.f };
	}
	// Go a small step backward (dir * 1e-3f), because in some cases the
	// scene boundary are true surface (e.g. cornell box), which would be
	// ignored/unshadowed in this case.
	return PositionSample{ bounds.min + position * sides - dir * 1e-3f, AreaPdf{ 1.f / area } };
}

CUDA_FUNCTION __forceinline__ float projected_area(const scene::Direction& dir,
												   const ei::Box& bounds) {
	ei::Vec3 sides = bounds.max - bounds.min;
	float projAx = sides.y * sides.z * ei::abs(dir.x);
	float projAy = sides.x * sides.z * ei::abs(dir.y);
	float projAz = sides.x * sides.y * ei::abs(dir.z);
	return projAx + projAy + projAz;
}

/*
 * A RndSet is a fixed size set of random numbers which may be consumed by a sampler.
 * There are different sets dependent on the application. Most samplers use 2 numbers.
 * For samplers with layer decisions (light tree, materials) there is a rndset with
 * an additional 64bit random value (RndSet2_1).
 */
struct RndSet2_1 {
	float u0;
	float u1;
	u64 i0;

	CUDA_FUNCTION __forceinline__ RndSet2_1(u64 x0, u64 x1) :
		u0{sample_uniform(u32(x0))},
		u1{sample_uniform(u32(x0 >> 32))},
		i0{x1}
	{}
};

struct RndSet2 {
	float u0;
	float u1;

	CUDA_FUNCTION __forceinline__ RndSet2(u64 x) :
		u0{sample_uniform(u32(x))},
		u1{sample_uniform(u32(x >> 32))}
	{}

	// Truncating initialization if someone asks for less than we have
	CUDA_FUNCTION __forceinline__ RndSet2(const RndSet2_1& x) :
		u0{x.u0}, u1{x.u1}
	{}
};

}} // namespace mufflon::math