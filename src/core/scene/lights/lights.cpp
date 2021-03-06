#include "lights.hpp"
#include "util/punning.hpp"
#include "core/math/rng.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/scene/materials/material_sampling.hpp"

namespace mufflon::scene::lights {

Spectrum get_flux(const PointLight& light) {
	return light.intensity * 4.f * ei::PI;
}


Spectrum get_flux(const SpotLight& light) {
	// Flux for the PBRT spotlight version with falloff ((cos(t)-cos(t_w))/(cos(t_f)-cos(t_w)))^4
	float cosFalloff = __half2float(light.cosFalloffStart);
	float cosWidth = __half2float(light.cosThetaMax);
	return light.intensity * (2.f * ei::PI * (1.f - cosFalloff) + (cosFalloff - cosWidth) / 5.f);
}


Spectrum get_flux(const AreaLightTriangle<Device::CPU>& light, const int* materials) {
	auto* mat = as<materials::MaterialDescriptorBase>(as<char>(materials) + materials[light.material]);
	float area = len(cross(light.posV[1u], light.posV[2u])) / 2.0f;
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen(util::pun<u32>(area));	// Use the area as seed
	Spectrum radianceSum{ 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		const ei::Vec2 bary = math::sample_barycentric(u.x, u.y);
		const UvCoordinate uv = light.uvV[0u] + light.uvV[1u] * bary.x + light.uvV[2u] * bary.y;
		materials::ParameterPack matParams;
		materials::fetch(*mat, uv, &matParams);
		radianceSum += materials::emission(matParams, Direction{0.0f, 0.0f, 1.0f}, Direction{0.0f, 0.0f, 1.0f}).value;
	}
	radianceSum /= 128;
	return radianceSum * area * 2 * ei::PI;
}


Spectrum get_flux(const AreaLightQuad<Device::CPU>& light, const int* materials) {
	auto* mat = as<materials::MaterialDescriptorBase>(as<char>(materials) + materials[light.material]);
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen((u32)util::pun<u64>(&light));	// Use different seeds per light
	Spectrum intensitySum { 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		const UvCoordinate uv = light.uvV[0u] + light.uvV[1u] * u.y + light.uvV[2u] * u.x + light.uvV[3u] * (u.x * u.y);
		const ei::Vec3 tangentX = light.posV[1u] + u.x * light.posV[3u];
		const ei::Vec3 tangentY = light.posV[2u] + u.y * light.posV[3u];
		const float area = len(cross(tangentY, tangentX));
		materials::ParameterPack matParams;
		materials::fetch(*mat, uv, &matParams);
		intensitySum += area * materials::emission(matParams, Direction{0.0f, 0.0f, 1.0f}, Direction{0.0f, 0.0f, 1.0f}).value;
	}
	intensitySum /= 128;
	return intensitySum * 2 * ei::PI;
}


Spectrum get_flux(const AreaLightSphere<Device::CPU>& light, const int* materials) {
	auto* mat = as<materials::MaterialDescriptorBase>(as<char>(materials) + materials[light.material]);
	float area = ei::surface(ei::Sphere{ light.position, light.radius });
	// Sample the radiance over the entire triangle region.
	math::GoldenRatio2D gen(util::pun<u32>(area));	// Use the area as seed
	Spectrum radianceSum{ 0.0f };
	for(int i = 0; i < 128; ++i) {// TODO: adaptive sample count?
		const ei::Vec2 u = math::sample_uniform(gen.next());
		// Get lon-lat, but in domain [0,1]
		float theta = acos(u.x * 2.0f - 1.0f) / ei::PI;
		float phi = u.y;
		materials::ParameterPack matParams;
		materials::fetch(*mat, UvCoordinate{phi, theta}, &matParams);
		radianceSum += materials::emission(matParams, Direction{0.0f, 0.0f, 1.0f}, Direction{0.0f, 0.0f, 1.0f}).value;
	}
	radianceSum /= 128;
	return radianceSum * area * 2 * ei::PI;
}


Spectrum get_flux(const DirectionalLight& light,
				  const ei::Vec3& aabbDiag) {
	mAssert(aabbDiag.x > 0 && aabbDiag.y > 0 && aabbDiag.z > 0);
	float surface = aabbDiag.y*aabbDiag.z*std::abs(light.direction.x)
		+ aabbDiag.x*aabbDiag.z*std::abs(light.direction.y)
		+ aabbDiag.x*aabbDiag.y*std::abs(light.direction.z);
	return light.irradiance * surface;
}

} // mufflon::scene::lights