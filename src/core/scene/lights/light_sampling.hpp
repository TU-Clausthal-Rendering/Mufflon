#pragma once

#include "lights.hpp"
#include "core/math/sampling.hpp"
#include "ei/vector.hpp"
#include "core/scene/types.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::lights {

struct Photon {
	math::PositionSample pos;
	math::DirectionSample dir;
};

struct NextEventEstimation {
	math::PositionSample pos;
	ei::Vec3 intensity;
	AngularPdf dirPdf;
};

/**
 * A RndSet is a fixed size set of random numbers which may be consumed by a light
 * sampler. There is currently no guarantee about their relative quality.
 */
struct RndSet {
	float u0;	// In [0,1)
	float u1;	// In [0,1)
	float u2;	// In [0,1)
	float u3;	// In [0,1)
};

// Sample a light source
inline __host__ __device__ __forceinline__ Photon sample_light(const PointLight& light,
															   const RndSet& rnd) {
	return Photon{ { light.position, AreaPdf(std::numeric_limits<float>::infinity()) },
				   math::sample_dir_sphere_uniform(rnd.u0, rnd.u1)};
}
inline __host__ __device__ __forceinline__ Photon sample_light(const SpotLight& light,
															   const RndSet& rnd) {
	// TODO: bring the falloff into it?
	float cosThetaMax = __half2float(light.cosThetaMax);
	return Photon{ { light.position, AreaPdf::infinite() },
				   math::sample_cone(rnd.u0, rnd.u1, cosThetaMax) };
}
inline __host__ __device__ __forceinline__ Photon sample_light(const AreaLightTriangle& light,
															   const RndSet& rnd) {
	ei::Triangle triangle = ei::Triangle(light.points[0u], light.points[1u],
											 light.points[2u]);
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	ei::Vec3 normal = ei::normalize(ei::cross(triangle.v2 - triangle.v0, triangle.v1 - triangle.v0));
	// TODO: can we assume cosine distribution?
	// TODO: how to transform into world space?
	return Photon{
		math::sample_position(triangle, rnd.u0, rnd.u1),
		math::sample_dir_cosine(rnd.u2, rnd.u3)
	};
}
inline __host__ __device__ __forceinline__ Photon sample_light(const AreaLightQuad& light,
															   const RndSet& rnd) {
	// Two-split decision: first select triangle, then use triangle sampling
	ei::Triangle first = ei::Triangle(light.points[0u], light.points[1u], light.points[2u]);
	ei::Triangle second = ei::Triangle(light.points[0u], light.points[2u], light.points[3u]);
	float areaFirst = ei::surface(first);
	float areaSecond = ei::surface(second);
	float split = areaFirst / (areaFirst + areaSecond);
	mAssert(!std::isnan(split));

	// TODO: can we assume cosine distribution?
	// TODO: how to transform into world space?

	if(rnd.u0 < split) {
		// Rescale the random number to be reusable
		float u0 = rnd.u0 * split;
		ei::Vec3 normal = ei::normalize(ei::cross(first.v2 - first.v0, first.v1 - first.v0));
		return Photon{
			math::sample_position(first, u0, rnd.u1),
			math::sample_dir_cosine(rnd.u2, rnd.u3)
		};
	} else {
		// Rescale the random number to be reusable
		float u0 = (rnd.u0 - split) / (1.f - split);
		ei::Vec3 normal = ei::normalize(ei::cross(second.v2 - second.v0, second.v1 - second.v0));
		return Photon{
			math::sample_position(second, u0, rnd.u1),
			math::sample_dir_cosine(rnd.u2, rnd.u3)
		};
	}
}
inline __host__ __device__ __forceinline__ Photon sample_light(const AreaLightSphere& light,
															   const RndSet& rnd) {
	math::DirectionSample dir = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	// TODO: convert into world space
	return Photon{
		math::PositionSample{ light.position + dir.direction * light.radius,
		  dir.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		math::sample_dir_cosine(rnd.u2, rnd.u3)
	};;
}
inline __host__ __device__ __forceinline__ Photon sample_light(const DirectionalLight& light,
															   const ei::Box& bounds,
															   const RndSet& rnd) {
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1, rnd.u2),
		math::DirectionSample{ light.direction, AngularPdf::infinite() }
	};
	// TODO
	return {};
}
template < Device dev >
inline __host__ __device__ __forceinline__ Photon sample_light(const EnvMapLight<dev>& light,
															   const RndSet& rnd) {
	// TODO
	return {};
}

} // namespace mufflon::scene::lights