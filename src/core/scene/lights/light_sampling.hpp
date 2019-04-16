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
	Point position;
	AreaPdf creationPdf;				// Pdf to create this connection event (depends on light choice probability and positional sampling)
	math::DirectionSample dir { Direction{0.0f}, AngularPdf{0.0f} };	// Direction to the connected vertex and the pdf to go into this direction when starting at the light
	float cosOut {0.0f};				// Cosine of the light-surface or 0 for non-hitable sources
	Spectrum diffIrradiance {0.0f};		// Unit: W/m²sr²
	float dist {0.0f};
	Direction geoNormal;
	float distSq {0.0f};
	//LightType type; // Not required ATM
};

// NEE distance will not go below this threshold (avoid infinity peaks
constexpr float DISTANCESQ_EPSILON = 1e-10f;


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
	const Spectrum flux = intensity * falloff / float(dir.pdf);
	return { {globalDir, dir.pdf}, flux };
}

// *** AREA LIGHT : TRIANGLE ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightTriangle<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// The normal of the triangle is implicit due to counter-clockwise vertex ordering
	ei::Vec3 normal = ei::cross(light.posV[1u], light.posV[2u]);
	const float area2 = len(normal);
	normal /= area2;
	const float area = area2 * 0.5f;
	// Sample barycentrics (we need position and uv at the same location)
	const ei::Vec2 bary = math::sample_barycentric(rnd.u0, rnd.u1);
	const ei::Vec3 position = light.posV[0u] + light.posV[1u] * bary.x + light.posV[2u] * bary.y;
	const UvCoordinate uv = light.uvV[0u] + light.uvV[1u] * bary.x + light.uvV[2u] * bary.y;
	const Spectrum scale = ei::unpackRGB9E5(light.scale);
	const Spectrum radiance = Spectrum{ sample(light.radianceTex, uv) } * scale;
	return Photon{
		{ position, AreaPdf{1.0f / area} },
		radiance * area,
		LightType::AREA_LIGHT_TRIANGLE,
		{normal, area}
	};
}

// *** AREA LIGHT : QUAD ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightQuad<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// The rnd coordinate is our uv.
	// Get the geometric normal. This requires an interpolation of the edges.
	const ei::Vec3 tangentX = light.posV[1u] + rnd.u0 * light.posV[3u];	// == lerp(e03, e12, u0)
	const ei::Vec3 tangentY = light.posV[2u] + rnd.u1 * light.posV[3u];	// == lerp(e01, e32, u1)
	ei::Vec3 normal = cross(tangentY, tangentX);
	const float area = len(normal);
	normal /= area;

	// The position is obtained by simple bilinear interpolation. To avoid
	// redundant computation we can use the intermediate results from the
	// normal computation.
	const ei::Vec3 position = light.posV[0u] + tangentX * rnd.u1 + light.posV[2u] * rnd.u0;
	// == const ei::Vec3 position = light.posV[0u] + light.posV[1u] * rnd.u1 + light.posV[2u] * rnd.u0 + light.posV[3u] * (rnd.u0 * rnd.u1);
	// The same goes for the uv coordinates
	const ei::Vec2 uv = light.uvV[0u] + light.uvV[1u] * rnd.u1 + light.uvV[2u] * rnd.u0 + light.uvV[3u] * (rnd.u0 * rnd.u1);

	const Spectrum radiance = Spectrum{ sample(light.radianceTex, uv) } * light.scale;

	return Photon{ { position, AreaPdf{ 1.0f / area } },
		radiance * area,
		LightType::AREA_LIGHT_QUAD,
		{normal, area} };
}

// *** AREA LIGHT : SPHERE ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const AreaLightSphere<CURRENT_DEV>& light,
													  const math::RndSet2& rnd) {
	// We don't need to convert the "normal" due to sphere symmetry
	const math::DirectionSample normal = math::sample_dir_sphere_uniform(rnd.u0, rnd.u1);
	UvCoordinate uvDummy;
	const Spectrum radiance = Spectrum{ sample(light.radianceTex, normal.direction, uvDummy) } * light.scale;
	const float area = 4 * ei::PI * light.radius * light.radius;
	return Photon{
		math::PositionSample{ light.position + normal.direction * light.radius,
							  normal.pdf.to_area_pdf(1.f, light.radius*light.radius) },
		radiance * area,
		LightType::AREA_LIGHT_SPHERE,
		{normal.direction, area}
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
	// flux = intensity * cosθ / pdf = intensity * π
	return { {globalDir, dir.pdf}, intensity * ei::PI };
}

// *** DIRECTIONAL LIGHT ***
CUDA_FUNCTION __forceinline__ Photon sample_light_pos(const DirectionalLight& light,
													  const ei::Box& bounds,
													  const math::RndSet2& rnd) {
	// TODO: invalid unit? irradiance != intensity != flux, photons should have flux...
	auto pos = math::sample_position(light.direction, bounds, rnd.u0, rnd.u1);
	Spectrum flux = light.irradiance / float(pos.pdf);
	return Photon{
		pos, flux, LightType::DIRECTIONAL_LIGHT,
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
		layer = sample.texel.x / texSize.x;
		sample.texel.x -= layer * texSize.x;
		// Bring the UV into the interval as well
		sample.uv.x = sample.uv.x * 6.0f - static_cast<float>(layer);
		dir.direction = textures::cubemap_uv_to_surface(sample.uv, layer);
		const float lsq = ei::lensq(dir.direction);
		const float l = ei::sqrt(lsq);
		dir.direction *= 1.f / l;
		dir.direction.z = -dir.direction.z;
		// See Johannes' renderer
		dir.pdf = AngularPdf(sample.pdf * lsq * l / 24.f);
	} else {
		// Spherical map
		// Convert UV to spherical...
		const float phi = sample.uv.x * 2.f * ei::PI;
		const float theta = (1.f - sample.uv.y) * ei::PI;
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
	auto pos = math::sample_position(-sample.dir.direction, bounds, rnd2.u0, rnd2.u1);;

	// Convert radiance to flux (applies division from Monte-Carlo)
	Spectrum flux = sample.radiance / float(pos.pdf) / float(sample.dir.pdf);

	return Photon { pos, flux, LightType::ENVMAP_LIGHT,
					Photon::SourceParam{-sample.dir.direction, sample.dir.pdf} };
}



// Connect to a light source
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const PointLight& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	ei::Vec3 direction = light.position - pos;
	const float distSq = lensq(direction) + DISTANCESQ_EPSILON;
	const float dist = sqrtf(distSq);
	direction /= dist;
	// Compute the contribution
	Spectrum diffIrradiance = light.intensity / distSq;
	return NextEventEstimation{
		light.position, AreaPdf::infinite(), {direction, AngularPdf{1.0f / (4 * ei::PI)}},
		0.0f, diffIrradiance, dist,
		Direction{0.0f}, distSq
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const SpotLight& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	ei::Vec3 direction = light.position - pos;
	const float distSq = lensq(direction) + DISTANCESQ_EPSILON;
	const float dist = sqrtf(distSq);
	direction /= dist;
	const math::EvalValue value = evaluate_spot(-direction, light.intensity,
										light.direction,
										light.cosThetaMax, light.cosFalloffStart);
	// Compute the contribution
	Spectrum diffIrradiance = value.value / distSq;
	return NextEventEstimation{
		light.position, AreaPdf::infinite(), {direction, value.pdf.forw},
		0.0f, diffIrradiance, dist,
		Direction{0.0f}, distSq
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightTriangle<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon photon = sample_light_pos(light, rnd);
	ei::Vec3 direction = photon.pos.position - pos;
	const float distSq = ei::lensq(direction) + DISTANCESQ_EPSILON;
	const float dist = sqrtf(distSq);
	direction /= dist;
	// Compute the contribution
	const math::EvalValue value = evaluate_area(-direction, photon.intensity,
										photon.source_param.area.normal);
	Spectrum diffIrradiance = value.value / distSq;
	return NextEventEstimation{
		photon.pos.position, photon.pos.pdf, {direction, value.pdf.forw},
		value.cosOut, diffIrradiance, dist,
		photon.source_param.area.normal, distSq
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightQuad<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	Photon photon = sample_light_pos(light, rnd);
	ei::Vec3 direction = photon.pos.position - pos;
	const float distSq = ei::lensq(direction) + DISTANCESQ_EPSILON;
	const float dist = sqrtf(distSq);
	direction /= dist;
	// Compute the contribution
	const math::EvalValue value = evaluate_area(-direction, photon.intensity,
										photon.source_param.area.normal);
	Spectrum diffIrradiance = value.value / distSq;
	return NextEventEstimation{
		photon.pos.position, photon.pos.pdf, {direction, value.pdf.forw},
		value.cosOut, diffIrradiance, dist,
		photon.source_param.area.normal, distSq
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const AreaLightSphere<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const math::RndSet2& rnd) {
	scene::Direction centerDir = pos - light.position;
	float dist = len(centerDir);
	if(dist <= light.radius) return NextEventEstimation{}; // Point inside
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
	const float cDistSq = lensq(connectionDir) + DISTANCESQ_EPSILON;
	const float cDist = sqrtf(cDistSq);
	connectionDir /= cDist;
	// Compute the contribution (diffIrradiance)
	UvCoordinate uvDummy;
	// TODO: instance rotation on globalDir?
	Spectrum radiance { sample(light.radianceTex, globalDir, uvDummy) };
	const float sampleArea = light.radius * light.radius / float(dir.pdf);
	const float cosOut = ei::max(0.0f, -dot(globalDir, connectionDir));
	radiance *= sampleArea / cDistSq;
	return NextEventEstimation{
		surfPos, AreaPdf{1.0f / sampleArea},
		{connectionDir, AngularPdf{cosOut / ei::PI}}, cosOut,
		radiance * light.scale,
		cDist, globalDir, cDistSq
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const DirectionalLight& light,
																const ei::Vec3& pos,
																const ei::Box& bounds) {
	// For Directional lights AngularPdf and AreaPdf are exchanged. This simplifies all
	// dependent code. The fictive connection point always has a connection distance
	// of MAX_SCENE_SIZE, independent of the real sampled position (orthographic projection
	// anyway). This makes it possible to convert the pdf in known ways at every
	// kind of event.
	AngularPdf posPdf { 1.0f / math::projected_area(light.direction, bounds) };
	return NextEventEstimation{
		pos - light.direction * len(bounds.max - bounds.min),
		AreaPdf::infinite(),	// Dummy pdf (the directional sampling pdf, converted)
		{-light.direction, posPdf}, 0.0f, light.irradiance, MAX_SCENE_SIZE,
		Direction{0.0f}, ei::sq(MAX_SCENE_SIZE),
	};
}
CUDA_FUNCTION __forceinline__ NextEventEstimation connect_light(const BackgroundDesc<CURRENT_DEV>& light,
																const ei::Vec3& pos,
																const ei::Box& bounds,
																const math::RndSet2& rnd) {
	auto sample = sample_light_dir(light, rnd.u0, rnd.u1);

	// See connect_light(DirectionalLight) for pdf argumentation.
	AngularPdf posPdf { 1.0f / math::projected_area(sample.dir.direction, bounds) };
	const Spectrum diffIrradiance = sample.radiance / float(sample.dir.pdf);
	return NextEventEstimation{
		pos + sample.dir.direction * len(bounds.max - bounds.min),
		sample.dir.pdf.to_area_pdf(1.0f, ei::sq(MAX_SCENE_SIZE)),
		{sample.dir.direction, posPdf}, 1.0f, diffIrradiance, MAX_SCENE_SIZE,
		Direction{0.0f}, ei::sq(MAX_SCENE_SIZE)
	};
}

// Evaluate a directional hit of the background.
// This function would be more logical in lights.hpp. But it requires textures
// and would increase header dependencies.
CUDA_FUNCTION math::EvalValue evaluate_background(const BackgroundDesc<CURRENT_DEV>& background,
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
				const float length = 1.0f / ei::max(ei::abs(direction));
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