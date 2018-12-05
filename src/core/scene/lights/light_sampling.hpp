#pragma once

#include "lights.hpp"
#include "core/export/api.h"
#include "ei/vector.hpp"
#include "ei/conversions.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/interface.hpp"
#include <math.h>
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace lights {

// Sampler result of positional light source sampling
struct Photon {
	math::PositionSample pos;
	Spectrum intensity;
	LightType type;
	// Deliver some additional information dependent on the type.
	// These are required to generate general purpose vertices.
	union SourceParam {
		CUDA_FUNCTION SourceParam() {}
		CUDA_FUNCTION SourceParam(const scene::Direction& d, half cT, half cFS) : spot{d, cT, cFS} {}
		CUDA_FUNCTION SourceParam(const scene::Direction& n, float a) : area{n, a} {}
		CUDA_FUNCTION SourceParam(const scene::Direction& d, AngularPdf p) : dir{d, p} {}
		struct {
			scene::Direction direction;
			half cosThetaMax;
			half cosFalloffStart;
		} spot;
		struct {
			scene::Direction normal;
			float area;		// TODO: remove? (ATM not used by the vertex)
		} area;
		struct {
			scene::Direction direction;
			AngularPdf dirPdf;
		} dir;
	} source_param;
};

// Sampler result of directional light source sampling
struct PhotonDir {
	math::DirectionSample dir;
	Spectrum flux;
};

struct NextEventEstimation {
	math::PositionSample pos;
	//math::DirectionSample dir;	// PDF not needed, because PT is the only user of NEE, maybe required later again??
	scene::Direction direction;		// From surface to the light source
	float cosOut;					// Cos of the surface or 0 for non-hitable sources
	Spectrum intensity;				// Unit: W/sr²
	float distSq;
	//LightType type; // Not required ATM
};


// Transform a direction from tangent into world space (convention Z-up vs. Y-up)
CUDA_FUNCTION __forceinline__ ei::Vec3 tangent2world(const ei::Vec3& dir,
													 const ei::Vec3& tangentX,
													 const ei::Vec3& tangentY,
													 const ei::Vec3& normal) {
	return dir.x * tangentY + dir.z * normal + dir.y * tangentY;
}

// Sample a light source
// *** POINT LIGHT ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const PointLight& light,
													  const math::RndSet2& rnd) {
	return Photon{ { light.position, AreaPdf::infinite() },
				   light.intensity, LightType::POINT_LIGHT };
}
CUDA_FUNCTION __forceinline__ PhotonDir sample_light_dir_point(const Spectrum& intensity,
															   const math::RndSet2& rnd) {
	return {math::sample_dir_sphere_uniform(rnd.u0, rnd.u1),
			intensity * 4 * ei::PI};
}

// *** SPOT LIGHT ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const SpotLight& light,
													  const math::RndSet2& rnd
) {
	return Photon{ { light.position, AreaPdf::infinite() },
				   light.intensity, LightType::SPOT_LIGHT,
				   {ei::unpackOctahedral32(light.direction), light.cosThetaMax, light.cosFalloffStart} };
}
CUDA_FUNCTION __forceinline__ PhotonDir sample_light_dir_spot(const Spectrum& intensity,
															  const scene::Direction direction,
															  half cosThetaMax,
															  half cosFalloffStart,
															  const math::RndSet2& rnd
) {
	// Sample direction in the cone
	math::DirectionSample dir = math::sample_cone_uniform(cosThetaMax, rnd.u0, rnd.u1);
	// Transform direction to world coordinates
	const scene::Direction tangentX = normalize(perpendicular(direction));
	const scene::Direction tangentY = cross(direction, tangentX);
	const scene::Direction globalDir = dir.direction.x * tangentX + dir.direction.y * tangentY + dir.direction.z * direction;
	// Compute falloff for cone
	const float falloff = scene::lights::get_falloff(dir.direction.z, cosThetaMax, cosFalloffStart);
	return { {globalDir, dir.pdf}, intensity * falloff }; // TODO flux likely wrong (misses the pdf?)
}

// *** AREA LIGHT : TRIANGLE ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightTriangle<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	const ei::Vec3 tangentX = light.points[1u] - light.points[0u];
	const ei::Vec3 tangentY = light.points[2u] - light.points[0u];
	ei::Vec3 normal = ei::cross(tangentX, tangentY);
	const float area2Inv = 1.0f / len(normal);
	normal *= area2Inv;
	// Sample barycentrics (we need position and uv at the same location)
	const ei::Vec2 bary = math::sample_barycentric(rnd.u0, rnd.u1);
	const ei::Vec3 position = light.points[0u] + tangentX * bary.x + tangentY * bary.y;
	const ei::Vec2 uv = light.uv[0u] + (light.uv[1u] - light.uv[0u]) * bary.x + (light.uv[2u] - light.uv[0u]) * bary.y;
	return Photon{
		{ position, AreaPdf{area2Inv * 2.0f} },
		Spectrum{ sample(light.radianceTex, uv) }, // TODO: radiance to intensity?
		LightType::AREA_LIGHT_TRIANGLE,
		{normal, 0.5f / float(area2Inv)}
	};
}

// *** AREA LIGHT : QUAD ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightQuad<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
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
	float u2;
	if(rnd.u0 < split) {
		// Rescale the random number to be reusable
		u2 = rnd.u0 / split;
		side = 0;
	} else {
		u2 = (rnd.u0 - split) / (1.f - split);
		side = 1;
	}

	// Sample the position on the selected triangle and account for chance to choose the triangle
	// Use barycentrics (we need position and uv at the same location)
	const ei::Vec2 bary = math::sample_barycentric(u2, rnd.u1);
	const ei::Vec3 position = light.points[0u] + (side ?
										tangent2X * bary.x + tangent2Y * bary.y :
										tangent1X * bary.x + tangent1Y * bary.y);
	// TODO: compute UVs based on inverted bilinar interpolation? This is what should
	// be done for high quality (here and in intersections with quads)
	const ei::Vec2 uv = light.uv[0u]
		+ (light.uv[side?2u:1u] - light.uv[0u]) * bary.x
		+ (light.uv[side?3u:2u] - light.uv[0u]) * bary.y;

	return Photon{ { position, AreaPdf{ 1.0f / (area1 + area2) } },
		Spectrum{sample(light.radianceTex, uv)}, // TODO: radiance to intensity?
		LightType::AREA_LIGHT_QUAD,
		{side ? normal2 / (area2 * 2.0f) : normal1 / (area1 * 2.0f), area1 + area2} };
}

// *** AREA LIGHT : SPHERE ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightSphere<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	const math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	const ei::Vec2 uv {atan2(normal.direction.y, normal.direction.x), acos(normal.direction.z)}; // TODO: not using the sampler (inline) would allow to avoid the atan function here
	// TODO: what is the outgoing size?
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
							  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		Spectrum{sample(light.radianceTex, uv)}, LightType::AREA_LIGHT_SPHERE,
		{normal.direction, 4*ei::PI*ei::sq(light.radius)}
	};
}

// *** AREA LIGHT : DIRECTION (ALL) ***
CUDA_FUNCTION __forceinline__ PhotonDir sample_light_dir_area(const Spectrum& intensity,
															  const scene::Direction& normal,
															  const math::RndSet2& rnd) {
	// Sample the direction (lambertian model)
	math::DirectionSample dir = math::sample_dir_cosine(rnd.u0, rnd.u1);
	// Transform direction to world coordinates
	const ei::Vec3 tangentX = normalize(perpendicular(normal));
	const ei::Vec3 tangentY = cross(normal, tangentX);
	const ei::Vec3 globalDir = dir.direction.x * tangentX + dir.direction.y * tangentY + dir.direction.z * normal;
	return { {globalDir, dir.pdf}, intensity * dir.direction.z};
}

// *** DIRECTIONAL LIGHT ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const DirectionalLight& light,
													  const ei::Box& bounds,
													  const math::RndSet2& rnd) {
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1),
		light.radiance, LightType::DIRECTIONAL_LIGHT,
		{light.direction, AngularPdf::infinite()}
	};
}
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const EnvMapLight<CURRENT_DEV>& light,
													  const ei::Box& bounds,
													  const math::RndSet2_1& rnd) {
	// TODO: sample direction from texture
	math::DirectionSample dir{};
	Spectrum envValue = light.flux;	// THIS IS JUST A DUMMY. Insert the value from the texture here
	// Sample a start position on the bounding box
	math::RndSet2 rnd2{rnd.i0};
	return Photon{
		math::sample_position(dir.direction, bounds, rnd2.u0, rnd2.u1),
		envValue, LightType::ENVMAP_LIGHT,
		{dir.direction, dir.pdf}
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
		direction, 0.0f, light.intensity, distSqr
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
		direction, 0.0f, light.intensity * falloff, distSqr
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightTriangle<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon posSample = sample_light_pos(light, rnd);
	ei::Vec3 direction = posSample.pos.position - pos;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	// Compute the differential irradiance and make sure we went out the right
	// direction.
	float cosOut = ei::max(0.0f, ei::dot(posSample.source_param.area.normal, direction));
	return NextEventEstimation{
		posSample.pos, direction, cosOut, posSample.intensity * cosOut, distSqr
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightQuad<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon posSample = sample_light_pos(light, rnd);
	ei::Vec3 direction = posSample.pos.position - pos;
	const float distSqr = ei::lensq(direction);
	direction /= sqrtf(distSqr);
	// Compute the differential irradiance and make sure we went out the right
	// direction.
	float cosOut = ei::max(0.0f, ei::dot(posSample.source_param.area.normal, direction));
	return NextEventEstimation{
		posSample.pos, direction, cosOut, posSample.intensity * cosOut, distSqr
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightSphere<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		scene::Direction{}, 0.0f,
		ei::Vec3{},
		float{}
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const DirectionalLight& light,
																const ei::Vec3& pos,
																const ei::Box& bounds,
																const math::RndSet2& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		scene::Direction{}, 0.0f,
		ei::Vec3{},
		float{}
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const EnvMapLight<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	// TODO
	return NextEventEstimation{
		math::PositionSample{},
		scene::Direction{}, 0.0f,
		ei::Vec3{},
		float{}
	};
}

}}} // namespace mufflon::scene::lights