#pragma once

#include "lights.hpp"
#include "texture_sampling.hpp"
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
	//math::PositionSample pos;		// Not required ATM
	//math::DirectionSample dir;	// PDF not needed, because PT is the only user of NEE, maybe required later again??
	scene::Direction direction;		// From surface to the light source, normalized
	float cosOut;					// Cos of the surface or 0 for non-hitable sources
	Spectrum diffIrradiance;		// Unit: W/m²sr²
	float dist;
	float distSq;
	AreaPdf creationPdf;			// Pdf to create this connection event (depends on light choice probability and positional sampling)
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
				   { light.direction, light.cosThetaMax, light.cosFalloffStart } };
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
	const Spectrum scale = ei::unpackRGB9E5(light.scale);
	return Photon{
		{ position, AreaPdf{area2Inv * 2.0f} },
		Spectrum{ sample(light.radianceTex, uv) } * scale, // TODO: radiance to intensity?
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

	const Spectrum scale = ei::unpackRGB9E5(light.scale);

	return Photon{ { position, AreaPdf{ 1.0f / (area1 + area2) } },
		Spectrum{sample(light.radianceTex, uv)} * scale, // TODO: radiance to intensity?
		LightType::AREA_LIGHT_QUAD,
		{side ? normal2 / (area2 * 2.0f) : normal1 / (area1 * 2.0f), area1 + area2} };
}

// *** AREA LIGHT : SPHERE ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightSphere<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	const math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	const Spectrum scale = ei::unpackRGB9E5(light.scale);
	UvCoordinate uvDummy;
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
							  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		Spectrum{sample(light.radianceTex, normal.direction, uvDummy)} * scale,
		LightType::AREA_LIGHT_SPHERE,
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
	// TODO: invalid unit? irradiance != intensity != flux, photons should have flux...
	return Photon{
		math::sample_position(light.direction, bounds, rnd.u0, rnd.u1),
		light.irradiance, LightType::DIRECTIONAL_LIGHT,
		{light.direction, AngularPdf::infinite()}
	};
}

// *** ENVMAP ***
// Samples a direction and evaluates the envmap's radiance in that direction
CUDA_FUNCTION struct { math::DirectionSample dir; Spectrum radiance; }
sample_light_dir(const BackgroundDesc<CURRENT_DEV>& light,
				 const float u0, const float u1) {
	if(light.type == BackgroundType::COLORED) return {};
	// TODO: sample other types of backgrounds too.

	// First we sample the envmap texel
	EnvmapSampleResult sample = importance_sample_texture(light.summedAreaTable, u0, u1);
	// Depending on the type of envmap we will sample a different layer of the texture
	const u16 layers = textures::get_texture_layers(light.envmap);
	int layer = 0u;
	math::DirectionSample dir{};
	if(layers == 6u) {
		// Cubemap: adjust the layer and texel
		const Pixel texSize = textures::get_texture_size(light.envmap);
		int layer = sample.texel.x / texSize.x;
		sample.texel.x -= layer * texSize.x;
		// Bring the UV into the interval as well
		sample.uv.x -= static_cast<float>(layer);
		// Turn the texel coordinates into UVs and remap from [0, 1] to [-1, 1]
		dir.direction = textures::cubemap_uv_to_surface(sample.uv, layer);
		const float lsq = ei::lensq(dir.direction);
		const float l = ei::sqrt(lsq);
		dir.direction *= 1.f / l;
		// See Johannes' renderer
		dir.pdf = AngularPdf(sample.pdf * lsq * l / 24.f);
	} else {
		// Spherical map
		// Convert UV to spherical...
		const float phi = sample.uv.x * 2.f * ei::PI;
		const float theta = sample.uv.y * ei::PI;
		// ...and then to cartesian
		const float sinTheta = sin(theta);
		const float cosTheta = cos(theta);
		const float sinPhi = sin(phi);
		const float cosPhi = cos(phi);
		dir.direction = ei::Vec3{
			sinTheta * cosPhi,
			cosTheta,
			sinTheta * sinPhi,
		};
		// PBRT p. 850
		dir.pdf = AngularPdf{ sample.pdf / (2.f * ei::PI * ei::PI * sinTheta) };
	}

	// Use always nearest sampling (otherwise the sampler is biased).
	ei::Vec3 radiance { textures::read(light.envmap, sample.texel, layer) };
	radiance *= light.color;
	return { dir, radiance };
}

CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const BackgroundDesc<CURRENT_DEV>& light,
													  const ei::Box& bounds,
													  const math::RndSet2_1& rnd) {
	auto sample = sample_light_dir(light, rnd.u0, rnd.u1);

	// Sample a start position on the bounding box
	math::RndSet2 rnd2{ rnd.i0 };
	auto pos = math::sample_position(sample.dir.direction, bounds, rnd2.u0, rnd2.u1);;

	// Convert radiance to flux (applies division from Monte-Carlo)
	Spectrum flux = sample.radiance / float(pos.pdf) / float(sample.dir.pdf);

	return Photon { pos, flux, LightType::ENVMAP_LIGHT,
					Photon::SourceParam{sample.dir.direction, sample.dir.pdf} };
}



// Connect to a light source
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const PointLight& light,
																const ei::Vec3& pos,
																const float distSqr,
																const math::RndSet2& rnd) {
	const float dist = sqrtf(distSqr);
	const ei::Vec3 direction = (light.position - pos) / dist;
	// Compute the contribution
	Spectrum diffIrradiance = light.intensity / distSqr;
	return NextEventEstimation{
		direction, 0.0f, diffIrradiance, dist, distSqr, AreaPdf::infinite()
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const SpotLight& light,
																const ei::Vec3& pos,
																const float distSqr,
																const math::RndSet2& rnd) {
	const float dist = sqrtf(distSqr);
	const ei::Vec3 direction = (light.position - pos) / dist;
	const math::EvalValue value = evaluate_spot(-direction, light.intensity,
										light.direction,
										light.cosThetaMax, light.cosFalloffStart);
	// Compute the contribution
	Spectrum diffIrradiance = value.value / distSqr;
	return NextEventEstimation{
		direction, value.cosOut, diffIrradiance, dist, distSqr, AreaPdf::infinite()
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightTriangle<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon posSample = sample_light_pos(light, rnd);
	ei::Vec3 direction = posSample.pos.position - pos;
	const float distSqr = ei::lensq(direction);
	const float dist = sqrtf(distSqr);
	direction /= dist;
	// Compute the contribution (TODO diffIrradiance)
	const math::EvalValue value = evaluate_area(-direction, posSample.intensity,
										posSample.source_param.area.normal);
	return NextEventEstimation{
		direction, value.cosOut, value.value, dist, distSqr, posSample.pos.pdf
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightQuad<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon posSample = sample_light_pos(light, rnd);
	ei::Vec3 direction = posSample.pos.position - pos;
	const float distSqr = ei::lensq(direction);
	const float dist = sqrtf(distSqr);
	direction /= dist;
	// Compute the contribution (TODO diffIrradiance)
	const math::EvalValue value = evaluate_area(-direction, posSample.intensity,
										posSample.source_param.area.normal);
	return NextEventEstimation{
		direction, value.cosOut, value.value, dist, distSqr, posSample.pos.pdf
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightSphere<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	scene::Direction centerDir = pos - light.position;
	float dist = len(centerDir);
	centerDir /= dist;
	// Compute the cosθ inside the sphere, to sample the solid angle extended by
	// the spherical cap.
	const float cosSphere = light.radius / dist;
	// Sample using the same method as spot lights
	const math::DirectionSample dir = math::sample_cone_uniform(cosSphere, rnd.u0, rnd.u1);
	// TODO: improve and use ei::base
	const scene::Direction tangentX = normalize(perpendicular(centerDir));
	const scene::Direction tangentY = cross(centerDir, tangentX);
	const scene::Direction globalDir = dir.direction.x * tangentX + dir.direction.y * tangentY + dir.direction.z * centerDir;
	// Now, connect to the point on the surface
	const scene::Point surfPos = light.position + light.radius * globalDir;
	scene::Direction connectionDir = surfPos - pos;
	const float cDistSq = lensq(connectionDir);
	const float cDist = sqrtf(cDistSq);
	connectionDir /= cDist;
	// Compute the contribution (diffIrradiance)
	UvCoordinate uvDummy;
	Spectrum radiance { sample(light.radianceTex, globalDir, uvDummy) };
	radiance *= light.radius * light.radius / (float(dir.pdf) * cDistSq);
	return NextEventEstimation{
		connectionDir, -dot(globalDir, connectionDir),
		radiance,
		cDist, cDistSq, dir.pdf.to_area_pdf(1.0f, ei::sq(light.radius))
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const DirectionalLight& light,
																const ei::Vec3& pos,
																const ei::Box& bounds) {
	AreaPdf posPdf { 1.0f / math::projected_area(light.direction, bounds) };
	// For Directional lights AngularPdf and AreaPdf are exchanged. This simplifies all
	// dependent code. The fictive connection point always has a connection distance
	// of MAX_SCENE_SIZE, independent of the real sampled position (orthographic projection
	// anyway). This makes it possible to convert the pdf in known ways at every
	// kind of event.
	return NextEventEstimation{
		-light.direction, 0.0f, light.irradiance, MAX_SCENE_SIZE, ei::sq(MAX_SCENE_SIZE),
		AreaPdf::infinite()	// Dummy pdf (the directional sampling pdf, converted)
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const BackgroundDesc<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const ei::Box& bounds,
																const math::RndSet2& rnd) {
	auto sample = sample_light_dir(light, rnd.u0, rnd.u1);

	// See connect_light(DirectionalLight) for pdf argumentation.
	AreaPdf posPdf { 1.0f / math::projected_area(sample.dir.direction, bounds) };
	const Spectrum diffIrradiance = sample.radiance / float(sample.dir.pdf);
	return NextEventEstimation{
		sample.dir.direction, 1.0f, diffIrradiance, MAX_SCENE_SIZE, ei::sq(MAX_SCENE_SIZE),
		sample.dir.pdf.to_area_pdf(1.0f, ei::sq(MAX_SCENE_SIZE))
	};
}

// Evaluate a directional hit of the background.
// This function would be more logical in lights.hpp. But it requires textures
// and would increase header dependencies.
template < Device dev >
CUDA_FUNCTION math::EvalValue evaluate_background(const BackgroundDesc<dev>& background,
												  const ei::Vec3& direction) {
	switch(background.type) {
		case BackgroundType::COLORED: return { background.color, 1.0f, AngularPdf{0.0f}, 
			AngularPdf{1.0f / (4.0f * ei::PI)} };
		case BackgroundType::ENVMAP: {
			UvCoordinate uv;
			Spectrum radiance { textures::sample(background.envmap, direction, uv) };
			// Get the p-value which was used to create the summed area table
			constexpr Spectrum LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
			// Get the integral from the table
			const Pixel texSize = textures::get_texture_size(background.summedAreaTable);
			const float cdf = textures::read(background.summedAreaTable, texSize - 1).x / (texSize.y * texSize.x);
			float pdfScale = dot(LUM_WEIGHT, radiance);
			// To complete the PDF we need the Jacobians of the map
			if(textures::get_texture_layers(background.envmap) == 6u) {
				// Cube map
				//ei::Vec3 projDir = direction / ei::max(direction);
				//pdfScale = powf(lensq(projDir), 1.5f) / 24.0f;
				// Should be equivalent to:
				const float length = 1.0f / ei::max(direction);
				pdfScale *= length * length * length / 24.0f;
			} else {
				// Polar map
				// The sin(θ) from luminance scale cancels out with the sin(θ)
				// from the Jacobian.
				const int pixelY = ei::floor(uv.y * texSize.y);
				const float sinPixel = sinf(ei::PI * static_cast<float>(pixelY + 0.5f) / static_cast<float>(texSize.y));
				const float sinJac = sqrtf(1.0f - direction.y * direction.y);
				pdfScale *= sinPixel / (2.0f * ei::PI * ei::PI * sinJac);
			}
			radiance *= background.color;
			return { radiance, 1.0f, AngularPdf{0.0f}, AngularPdf{pdfScale / cdf} };
		}
		default: mAssert(false); return {};
	}
}

}}} // namespace mufflon::scene::lights