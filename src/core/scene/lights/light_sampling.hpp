#pragma once

#include "lights.hpp"
#include "core/math/sampling.hpp"
#include "ei/vector.hpp"
#include "ei/conversions.hpp"
#include "core/scene/types.hpp"
#include "export/api.hpp"
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

// Transform a direction from tangent into world space (convention Z-up vs. Y-up)
CUDA_FUNCTION __forceinline__ ei::Vec3 tangent2world(const ei::Vec3& dir,
													 const ei::Vec3& tangentX,
													 const ei::Vec3& tangentY,
													 const ei::Vec3& normal) {
	return ei::Vec3{ dir.x * tangentY, dir.z * normal, dir.y * tangentY };
}

// Sample a light source
CUDA_FUNCTION __forceinline__ Photon sample_light(const PointLight& light,
															   const RndSet& rnd) {
	return Photon{ { light.position, AreaPdf(std::numeric_limits<float>::infinity()) },
				   math::sample_dir_sphere_uniform(rnd.u0, rnd.u1)};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const SpotLight& light,
															   const RndSet& rnd) {
	float cosThetaMax = __half2float(light.cosThetaMax);
	// Sample direction in the cone
	math::DirectionSample dir = math::sample_cone(rnd.u0, rnd.u1, cosThetaMax);
	// Transform direction to world coordinates
	// For that we need an arbitrary "up"-vector to compute our two tangents
	ei::Vec3 up{ 0, 1, 0 };
	if(fabsf(ei::dot(up, dir.direction)) < 0.05f) {
		// Too close to our up direction -> take "random" other vector
		up = ei::Vec3{ 1, 0, 0 };
	}
	// Compute tangent space
	ei::Vec3 tangentX = ei::normalize(ei::cross(up, dir.direction));
	ei::Vec3 tangentY = ei::cross(dir.direction, tangentX);
	dir.direction = tangent2world(dir.direction, tangentX, tangentY,
								  ei::unpackOctahedral32(light.direction));

	return Photon{ { light.position, AreaPdf::infinite() },
				   dir };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightTriangle& light,
															   const RndSet& rnd) {
	ei::Triangle triangle = ei::Triangle(light.points[0u], light.points[1u],
											 light.points[2u]);
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	ei::Vec3 tangentX = triangle.v2 - triangle.v0;
	ei::Vec3 tangentY = triangle.v2 - triangle.v0;
	ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Sample the direction (lambertian model)
	math::DirectionSample dir = math::sample_dir_cosine(rnd.u2, rnd.u3);
	// Transform into world space (Z-up to Y-up)
	dir.direction = tangent2world(dir.direction, tangentY, tangentY, normal);
	return Photon{
		math::sample_position(triangle, rnd.u0, rnd.u1),
		dir
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightQuad& light,
															   const RndSet& rnd) {
	// Two-split decision: first select triangle, then use triangle sampling
	ei::Triangle first = ei::Triangle(light.points[0u], light.points[1u], light.points[2u]);
	ei::Triangle second = ei::Triangle(light.points[0u], light.points[2u], light.points[3u]);
	float areaFirst = ei::surface(first);
	float areaSecond = ei::surface(second);
	float split = areaFirst / (areaFirst + areaSecond);
	mAssert(!std::isnan(split));

	// Decide what side we're on
	AreaPdf pdf;
	float u0;
	ei::Triangle* side;
	if(rnd.u0 < split) {
		// Rescale the random number to be reusable
		u0 = rnd.u0 * split;
		pdf = AreaPdf{ split };
		side = &first;
	} else {
		pdf = AreaPdf{ 1.f - split };
		u0 = (rnd.u0 - split) / (1.f - split);
		side = &second;
	}

	ei::Vec3 tangentX = side->v2 - side->v0;
	ei::Vec3 tangentY = side->v1 - side->v0;
	ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Sample the position on the selected triangle and account for chance to choose the triangle
	math::PositionSample pos = math::sample_position(*side, u0, rnd.u1);
	pos.pdf *= pdf;
	// Transform direction to world coordinates
	math::DirectionSample dir = math::sample_dir_cosine(rnd.u2, rnd.u3);
	dir.direction = tangent2world(dir.direction, tangentX, tangentY, normal);
	return Photon{ pos, dir };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightSphere& light,
															   const RndSet& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
		  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		math::sample_dir_cosine(rnd.u2, rnd.u3)
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const DirectionalLight& light,
															   const ei::Box& bounds,
															   const RndSet& rnd) {
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1, rnd.u2),
		math::DirectionSample{ light.direction, AngularPdf::infinite() }
	};
}
template < Device dev >
CUDA_FUNCTION __forceinline__ Photon sample_light(const EnvMapLight<dev>& light,
															   const RndSet& rnd) {
	(void)light;
	(void)rnd;
	// TODO
	return {};
}

} // namespace mufflon::scene::lights