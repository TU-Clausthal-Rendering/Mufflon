#pragma once

#include "lights.hpp"
#include "export/api.hpp"
#include "ei/vector.hpp"
#include "ei/conversions.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/interface.hpp"
#include <math.h>
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace lights {

struct Photon {
	math::PositionSample pos;
	Spectrum intensity;
	LightType type;
	// Deliver some additional information dependent on the type.
	// These are required to generate general purpose vertices.
	union SourceParam {
		CUDA_FUNCTION SourceParam() {}
		struct {
			ei::Vec3 direction;
			half cosThetaMax;
			half cosFalloffStart;
		} spot;
		struct {
			ei::Vec3 normal;
			float area;		// TODO: remove? (ATM not used by the vertex)
		} area;
		struct {
			ei::Vec3 direction;
			AngularPdf dirPdf;
		} dir;
	} source_param;
};

struct NextEventEstimation {
	math::PositionSample pos;
	math::DirectionSample dir;
	Spectrum diffIrradiance; // Unit: W/m²
	float distSqr;
	LightType type;
};

// INTERNAL: only used from 'private' methods
struct SurfaceSample {
	math::PositionSample pos;
	UvCoordinate uv;
	scene::Direction normal;
};

// Transform a direction from tangent into world space (convention Z-up vs. Y-up)
CUDA_FUNCTION __forceinline__ ei::Vec3 tangent2world(const ei::Vec3& dir,
													 const ei::Vec3& tangentX,
													 const ei::Vec3& tangentY,
													 const ei::Vec3& normal) {
	return dir.x * tangentY + dir.z * normal + dir.y * tangentY;
}

// Sample a light source
CUDA_FUNCTION __forceinline__ Photon sample_light(const PointLight& light,
												  const math::RndSet2& rnd) {
	return Photon{ { light.position, AreaPdf::infinite() },
				   light.intensity, LightType::POINT_LIGHT };
}

CUDA_FUNCTION __forceinline__ Photon sample_light(const SpotLight& light,
												  const math::RndSet2& rnd
) {
	return Photon{ { light.position, AreaPdf::infinite() },
				   light.intensity, LightType::SPOT_LIGHT };
}

CUDA_FUNCTION __forceinline__ SurfaceSample
sample_light_pos(const AreaLightTriangle<CURRENT_DEV>& light, float u0, float u1) {
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	const ei::Vec3 tangentX = light.points[1u] - light.points[0u];
	const ei::Vec3 tangentY = light.points[2u] - light.points[0u];
	ei::Vec3 normal = ei::cross(tangentX, tangentY);
	const float area2Inv = 1.0f / len(normal);
	normal *= area2Inv;
	// Sample barycentrics (we need position and uv at the same location)
	const ei::Vec2 bary = math::sample_barycentric(u0, u1);
	const ei::Vec3 position = light.points[0u] + tangentX * bary.x + tangentY * bary.y;
	const ei::Vec2 uv = light.uv[0u] + (light.uv[1u] - light.uv[0u]) * bary.x + (light.uv[2u] - light.uv[0u]) * bary.y;
	return { math::PositionSample{position, AreaPdf{ area2Inv * 2.0f }},
			 uv, normal };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightTriangle<CURRENT_DEV>& light,
												  const math::RndSet2& rnd) {
	SurfaceSample posSample = sample_light_pos(light, rnd.u0, rnd.u1);
	// TODO: what is the outgoing size?
	return Photon{
		posSample.pos, Spectrum{ sample(light.radianceTex, posSample.uv) },
		LightType::AREA_LIGHT_TRIANGLE
	};
}

CUDA_FUNCTION __forceinline__ SurfaceSample
sample_light_pos(const AreaLightQuad<CURRENT_DEV>& light, float u0, float u1) {
	// Two-split decision: first select triangle, then use triangle sampling.
	// Try to hold things in registers by conditional moves later on.
	const ei::Vec3 tangent1X = light.points[2u] - light.points[0u];
	const ei::Vec3 tangent1Y = light.points[1u] - light.points[0u];
	const ei::Vec3 normal1 = cross(tangent1X, tangent1Y);
	const ei::Vec3 tangent2X = light.points[3u] - light.points[0u];
	const ei::Vec3 tangent2Y = light.points[2u] - light.points[0u];
	const ei::Vec3 normal2 = cross(tangent1X, tangent1Y);
	const float area1 = len(normal1) * 0.5f;
	const float area2 = len(normal2) * 0.5f;
	const float split = area1 / (area1 + area2);
	mAssert(!isnan(split));
	// TODO: storing split could speed up things a lot (only need tangents/len of one triangle)

	// Decide what side we're on
	int side;
	if(u0 < split) {
		// Rescale the random number to be reusable
		u0 = u0 / split;
		side = 0;
	} else {
		u0 = (u0 - split) / (1.f - split);
		side = 1;
	}

	// Sample the position on the selected triangle and account for chance to choose the triangle
	// Use barycentrics (we need position and uv at the same location)
	const ei::Vec2 bary = math::sample_barycentric(u0, u1);
	const ei::Vec3 position = light.points[0u] + (side ?
										tangent2X * bary.x + tangent2Y * bary.y :
										tangent1X * bary.x + tangent1Y * bary.y);
	// TODO: compute UVs based on inverted bilinar interpolation? This is what should
	// be done for high quality (here and in intersections with quads)
	const ei::Vec2 uv = light.uv[0u]
		+ (light.uv[side?2u:1u] - light.uv[0u]) * bary.x
		+ (light.uv[side?3u:2u] - light.uv[0u]) * bary.y;
	return { math::PositionSample{position, AreaPdf{ 1.0f / (area1 + area2) }}, uv,
			 side ? normal2 / (area2 * 2.0f) : normal1 / (area1 * 2.0f) };
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightQuad<CURRENT_DEV>& light,
												  const math::RndSet2& rnd) {
	SurfaceSample posSample = sample_light_pos(light, rnd.u0, rnd.u1);
	// TODO: what is the outgoing size?
	return Photon{ posSample.pos,
		Spectrum{sample(light.radianceTex, posSample.uv)}, LightType::AREA_LIGHT_QUAD };
}

CUDA_FUNCTION __forceinline__ Photon sample_light(const AreaLightSphere<CURRENT_DEV>& light,
												  const math::RndSet2& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	const math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	const ei::Vec2 uv {atan2(normal.direction.y, normal.direction.x), acos(normal.direction.z)}; // TODO: not using the sampler (inline) would allow to avoid the atan function here
	// TODO: what is the outgoing size?
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
							  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		Spectrum{sample(light.radianceTex, uv)}, LightType::AREA_LIGHT_SPHERE
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const DirectionalLight& light,
												  const ei::Box& bounds,
												  const math::RndSet2& rnd) {
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1),
		light.radiance, LightType::DIRECTIONAL_LIGHT
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light(const EnvMapLight<CURRENT_DEV>& light,
												  const math::RndSet2& rnd) {
	(void)light;
	(void)rnd;
	// TODO
	return Photon{
		{},{},
		LightType::ENVMAP_LIGHT
	};
}

// Connect to a light source
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const PointLight& light,
																const ei::Vec3& pos,
																const float distSqr,
																const math::RndSet2& rnd) {
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
																const math::RndSet2& rnd) {
	float cosThetaMax = __half2float(light.cosThetaMax);
	const ei::Vec3 direction = (light.position - pos) / sqrtf(distSqr);
	const float falloff = get_falloff(-ei::dot(ei::unpackOctahedral32(light.direction),
											   direction),
									  cosThetaMax, __half2float(light.cosFalloffStart));
	return NextEventEstimation{
		math::PositionSample{ light.position, AreaPdf::infinite() },
		math::DirectionSample{ direction, math::get_uniform_cone_pdf(cosThetaMax) },
		light.intensity * falloff / distSqr, distSqr, LightType::SPOT_LIGHT
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightTriangle<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	SurfaceSample posSample = sample_light_pos(light, rnd.u0, rnd.u1);
	ei::Vec3 direction = pos - posSample.pos.position;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	// Compute the differential irradiance and make sure we went out the right
	// direction.
	float cosOut = ei::dot(posSample.normal, direction);
	Spectrum radiance{ sample(light.radianceTex, posSample.uv) };
	const ei::Vec3 diffIrradiance = (cosOut > 0u)
										? (radiance * cosOut / (distSqr * float(posSample.pos.pdf)))
										: ei::Vec3{ 0 };
	return NextEventEstimation{
		posSample.pos,
		math::DirectionSample{ direction, math::get_cosine_dir_pdf(cosOut)},
		diffIrradiance, distSqr, LightType::AREA_LIGHT_TRIANGLE
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightQuad<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	SurfaceSample posSample = sample_light_pos(light, rnd.u0, rnd.u1);
	ei::Vec3 direction = pos - posSample.pos.position;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	// Compute the differential irradiance and make sure we went out the right
	// direction.
	float cosOut = ei::dot(posSample.normal, direction);
	Spectrum radiance{ sample(light.radianceTex, posSample.uv) };
	const ei::Vec3 diffIrradiance = (cosOut > 0u)
										? (radiance * cosOut / (distSqr * float(posSample.pos.pdf)))
										: ei::Vec3{ 0 };
	return NextEventEstimation{
		posSample.pos, math::DirectionSample{ direction,
											  math::get_cosine_dir_pdf(cosOut)},
		diffIrradiance, distSqr, LightType::AREA_LIGHT_QUAD
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightSphere<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
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
																const math::RndSet2& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		math::DirectionSample{},
		ei::Vec3{},
		float{}, LightType::DIRECTIONAL_LIGHT
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const EnvMapLight<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		math::DirectionSample{},
		ei::Vec3{},
		float{}, LightType::ENVMAP_LIGHT
	};
}

}}} // namespace mufflon::scene::lights