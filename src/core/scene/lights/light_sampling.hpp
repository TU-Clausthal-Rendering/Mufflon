#pragma once

#include "lights.hpp"
#include "export/api.hpp"
#include "ei/vector.hpp"
#include "ei/conversions.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include <math.h>
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace lights {

struct Photon {
	math::PositionSample pos;
	math::DirectionSample dir;
	ei::Vec3 intensity;
	LightType type;
};

struct NextEventEstimation {
	math::PositionSample pos;
	math::DirectionSample dir;
	ei::Vec3 diffIrradiance; // Unit: W/m²
	float distSqr;
	LightType type;
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
	return dir.x * tangentY + dir.z * normal + dir.y * tangentY;
}

// Computes the falloff of a spotlight
CUDA_FUNCTION __forceinline__ float get_falloff(const float cosTheta,
												const float cosThetaMax,
												const float cosFalloffStart) {
	if(cosTheta >= cosThetaMax) {
		if(cosTheta >= cosFalloffStart)
			return 1.f;
		else
			return powf((cosTheta - cosThetaMax) / (cosFalloffStart - cosThetaMax), 4u);
	}
	return 0.f;
}

// Sample a light source
CUDA_FUNCTION __forceinline__ Photon sample_light(const PointLight& light,
												  const RndSet& rnd) {
	return Photon{ { light.position, AreaPdf::infinite() },
				   math::sample_dir_sphere_uniform(rnd.u0, rnd.u1),
				   light.intensity, LightType::POINT_LIGHT };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const SpotLight& light,
												  const RndSet& rnd) {
	const float cosThetaMax = __half2float(light.cosThetaMax);
	const float cosFalloffStart = __half2float(light.cosFalloffStart);
	// Sample direction in the cone
	math::DirectionSample dir = math::sample_cone_uniform(rnd.u0, rnd.u1, cosThetaMax);
	// Transform direction to world coordinates
	// For that we need an arbitrary "up"-vector to compute our two tangents
	ei::Vec3 up{ 0, 1, 0 };
	if(fabsf(ei::dot(up, dir.direction)) < 0.05f) {
		// Too close to our up direction -> take "random" other vector
		up = ei::Vec3{ 1, 0, 0 };
	}
	// Compute tangent space
	const ei::Vec3 tangentX = ei::normalize(ei::cross(up, dir.direction));
	const ei::Vec3 tangentY = ei::cross(dir.direction, tangentX);
	dir.direction = tangent2world(dir.direction, tangentX, tangentY,
								  ei::unpackOctahedral32(light.direction));
	// Compute falloff for cone
	const float falloff = get_falloff(ei::dot(ei::unpackOctahedral32(light.direction), dir.direction),
								cosThetaMax, cosFalloffStart);

	return Photon{ { light.position, AreaPdf::infinite() },
				   dir, light.intensity * falloff, LightType::SPOT_LIGHT };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightTriangle& light,
												  const RndSet& rnd) {
	const ei::Triangle triangle = ei::Triangle(light.points[0u], light.points[1u],
										 light.points[2u]);
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	const ei::Vec3 tangentX = triangle.v2 - triangle.v0;
	const ei::Vec3 tangentY = triangle.v1 - triangle.v0;
	const ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Sample the direction (lambertian model)
	math::DirectionSample dir = math::sample_dir_cosine(rnd.u2, rnd.u3);
	// Transform into world space (Z-up to Y-up)
	dir.direction = tangent2world(dir.direction, tangentY, tangentY, normal);
	// TODO: what is the outgoing size?
	return Photon{
		math::sample_position(triangle, rnd.u0, rnd.u1),
		dir, light.radiance,
		LightType::AREA_LIGHT_TRIANGLE
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightQuad& light,
												  const RndSet& rnd) {
	// Two-split decision: first select triangle, then use triangle sampling
	const ei::Triangle first = ei::Triangle(light.points[0u], light.points[1u], light.points[2u]);
	const ei::Triangle second = ei::Triangle(light.points[0u], light.points[2u], light.points[3u]);
	const float areaFirst = ei::surface(first);
	const float areaSecond = ei::surface(second);
	const float split = areaFirst / (areaFirst + areaSecond);
	mAssert(!isnan(split));

	// Decide what side we're on
	AreaPdf pdf;
	float u0;
	const ei::Triangle* side;
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

	const ei::Vec3 tangentX = side->v2 - side->v0;
	const ei::Vec3 tangentY = side->v1 - side->v0;
	const ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Sample the position on the selected triangle and account for chance to choose the triangle
	math::PositionSample pos = math::sample_position(*side, u0, rnd.u1);
	pos.pdf *= pdf;
	// Transform direction to world coordinates
	math::DirectionSample dir = math::sample_dir_cosine(rnd.u2, rnd.u3);
	dir.direction = tangent2world(dir.direction, tangentX, tangentY, normal);
	// TODO: what is the outgoing size?
	return Photon{ pos, dir, light.radiance, LightType::AREA_LIGHT_QUAD };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightSphere& light,
												  const RndSet& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	const math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	// TODO: what is the outgoing size?
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
							  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		math::sample_dir_cosine(rnd.u2, rnd.u3),
		light.radiance, LightType::AREA_LIGHT_SPHERE
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const DirectionalLight& light,
												  const ei::Box& bounds,
												  const RndSet& rnd) {
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1, rnd.u2),
		math::DirectionSample{ light.direction, AngularPdf::infinite() },
		light.radiance, LightType::DIRECTIONAL_LIGHT
	};
}
template < Device dev >
CUDA_FUNCTION __forceinline__ Photon sample_light(const EnvMapLight<dev>& light,
												  const RndSet& rnd) {
	(void)light;
	(void)rnd;
	// TODO
	return Photon{
		{},{},
		{}, LightType::ENVMAP_LIGHT
	};
}

// Connect to a light source
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const PointLight& light,
																const ei::Vec3& pos,
																const float distSqr,
																const RndSet& rnd) {
	const ei::Vec3 direction = (light.position - pos) / sqrtf(distSqr);
	return NextEventEstimation{
		math::PositionSample{ light.position, AreaPdf::infinite() },
		math::DirectionSample{ direction, math::get_uniform_dir_pdf() },
		light.intensity / distSqr, distSqr, LightType::POINT_LIGHT
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const SpotLight& light,
																const ei::Vec3& pos,
																const float distSqr,
																const RndSet& rnd) {
	float cosThetaMax = __half2float(light.cosThetaMax);
	const ei::Vec3 direction = (light.position - pos) / sqrtf(distSqr);
	const float falloff = get_falloff(ei::dot(ei::unpackOctahedral32(light.direction),
											  direction),
									  cosThetaMax, __half2float(light.cosFalloffStart));
	return NextEventEstimation{
		math::PositionSample{ light.position, AreaPdf::infinite() },
		math::DirectionSample{ direction, math::get_uniform_cone_pdf(cosThetaMax) },
		light.intensity * falloff / distSqr, distSqr, LightType::SPOT_LIGHT
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightTriangle& light,
																const ei::Vec3& pos,
																const RndSet& rnd) {
	const ei::Triangle triangle{ light.points[0u], light.points[1u], light.points[2u] };
	const math::PositionSample posSample = math::sample_position(triangle, rnd.u0, rnd.u1);
	ei::Vec3 direction = pos - posSample.position;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	const ei::Vec3 tangentX = triangle.v2 - triangle.v0;
	const ei::Vec3 tangentY = triangle.v1 - triangle.v0;
	const ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Compute the differential irradiance and make sure we went out the right
	// direction TODO is the formula correct?
	const ei::Vec3 diffIrradiance = (ei::dot(normal, direction) > 0u)
										? (light.radiance * ei::surface(triangle) / distSqr)
										: ei::Vec3{ 0 };
	return NextEventEstimation{
		posSample, math::DirectionSample{ direction,
										  math::get_cosine_dir_pdf(ei::dot(normal, direction))},
		diffIrradiance, distSqr, LightType::AREA_LIGHT_TRIANGLE
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightQuad& light,
																const ei::Vec3& pos,
																const RndSet& rnd) {
	// TODO: rejection sampling for quad side?
	const ei::Triangle first = ei::Triangle(light.points[0u], light.points[1u], light.points[2u]);
	const ei::Triangle second = ei::Triangle(light.points[0u], light.points[2u], light.points[3u]);
	const float areaFirst = ei::surface(first);
	const float areaSecond = ei::surface(second);
	const float split = areaFirst / (areaFirst + areaSecond);
	mAssert(!isnan(split));

	// Decide what side we're on
	AreaPdf pdf;
	float area;
	const ei::Triangle* side;
	if(rnd.u0 < split) {
		pdf = AreaPdf{ split };
		area = areaFirst;
		side = &first;
	} else {
		pdf = AreaPdf{ 1.f - split };
		area = areaSecond;
		side = &second;
	}

	math::PositionSample posSample = math::sample_position(*side, rnd.u2, rnd.u3);
	posSample.pdf *= pdf;
	ei::Vec3 direction = pos - posSample.position;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	const ei::Vec3 tangentX = side->v2 - side->v0;
	const ei::Vec3 tangentY = side->v1 - side->v0;
	const ei::Vec3 normal = ei::normalize(ei::cross(tangentX, tangentY));
	// Compute the differential irradiance and make sure we went out the right
	// direction TODO is the formula correct?
	const ei::Vec3 diffIrradiance = (ei::dot(normal, direction) > 0u)
										? (light.radiance * area / distSqr)
										: ei::Vec3{ 0 };
	return NextEventEstimation{
		posSample, math::DirectionSample{ direction,
										  math::get_cosine_dir_pdf(ei::dot(normal, direction))},
		diffIrradiance, distSqr, LightType::AREA_LIGHT_QUAD
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightSphere& light,
																const ei::Vec3& pos,
																const RndSet& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		math::DirectionSample{},
		ei::Vec3{},
		float{}, LightType::AREA_LIGHT_SPHERE
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const DirectionalLight& light,
																const ei::Vec3& pos,
																const ei::Box& bounds,
																const RndSet& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		math::DirectionSample{},
		ei::Vec3{},
		float{}, LightType::DIRECTIONAL_LIGHT
	};
}
template < Device dev >
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const EnvMapLight<dev>& light,
																const ei::Vec3& pos,
																const RndSet& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		math::DirectionSample{},
		ei::Vec3{},
		float{}, LightType::ENVMAP_LIGHT
	};
}

}}} // namespace mufflon::scene::lights