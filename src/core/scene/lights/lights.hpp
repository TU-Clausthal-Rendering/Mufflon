#pragma once

#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/math/sampling.hpp"
#include <cuda_fp16.h>
#include <type_traits>

#ifndef __CUDACC__
#include <variant>
#endif // __CUDACC__

namespace mufflon { namespace scene { namespace lights {

enum class LightType : u16 {
	POINT_LIGHT,
	SPOT_LIGHT,
	AREA_LIGHT_TRIANGLE,
	AREA_LIGHT_QUAD,
	AREA_LIGHT_SPHERE,
	DIRECTIONAL_LIGHT,
	ENVMAP_LIGHT,
	NUM_LIGHTS
};

// Important: light structs need to be packed to save storage space on the GPU
#pragma pack(push, 1)

/**
 * Structure representing a point light.
 * TODO: for measured light sources, we'd need to add a texture handle.
 */
struct alignas(16) PointLight {
	alignas(16) ei::Vec3 position;
	alignas(16) ei::Vec3 intensity;
};

/**
 * A spot light, following PBRT's description.
 * To save storage space, the direction is encoded in a single u32.
 * Additionally, the cosine terms for opening angle and falloff
 * are encoded 
 */
struct alignas(16) SpotLight {
	ei::Vec3 position;
	u32 direction;			// points away from the light source (direction of ligt flow), packed with ei::packOctahedral32
	ei::Vec3 intensity;
	half cosThetaMax;
	half cosFalloffStart;
};

/**
 * Area lights. One for every type of geometric primitive.
 * TODO: use an indirection to the primitive -> 16 Byte per light instead of up to 96.
 *		The cost of the indirection will be a more expensive fetch.
 * TODO: also test uv-compression (shared exponent or fixed point?)
 */
template < Device dev >
struct alignas(16) AreaLightTriangle {
	alignas(16) ei::Vec3 points[3u];
	alignas(8) UvCoordinate uv[3u];
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;
};
template < Device dev >
struct alignas(16) AreaLightQuad {
	ei::Vec3 points[4u];
	alignas(8) UvCoordinate uv[4u];
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;
};
template < Device dev >
struct alignas(16) AreaLightSphere {
	ei::Vec3 position;
	float radius;
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;
};

struct alignas(16) AreaLightTriangleDesc {
	alignas(16) ei::Vec3 points[3u];
	alignas(8) UvCoordinate uv[3u];
	alignas(8) TextureHandle radianceTex;
};
struct alignas(16) AreaLightQuadDesc {
	ei::Vec3 points[4u];
	alignas(8) UvCoordinate uv[4u];
	alignas(8) TextureHandle radianceTex;
};
struct alignas(16) AreaLightSphereDesc {
	ei::Vec3 position;
	float radius;
	alignas(8) TextureHandle radianceTex;
};

/**
 * Directional light. Doesn't have a position.
 */
struct alignas(16) DirectionalLight {
	alignas(16) ei::Vec3 direction;
	alignas(16) ei::Vec3 radiance;
};

/**
 * Environment-map light.
 */
template < Device dev >
struct alignas(16) EnvMapLight {
	textures::ConstTextureDevHandle_t<dev> texHandle;
	textures::ConstTextureDevHandle_t<dev> summedAreaTable;
	ei::Vec3 flux;
};

// Asserts to make sure the compiler actually followed our orders
static_assert(sizeof(PointLight) == 32 && alignof(PointLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(SpotLight) == 32 && alignof(SpotLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightTriangle<Device::CPU>) == 80 && alignof(AreaLightTriangle<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightTriangle<Device::CUDA>) == 80 && alignof(AreaLightTriangle<Device::CUDA>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightQuad<Device::CPU>) == 96 && alignof(AreaLightQuad<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightQuad<Device::CUDA>) == 96 && alignof(AreaLightQuad<Device::CUDA>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightSphere<Device::CPU>) == 32 && alignof(AreaLightSphere<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightSphere<Device::CUDA>) == 32 && alignof(AreaLightSphere<Device::CUDA>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(DirectionalLight) == 32 && alignof(DirectionalLight) == 16,
			  "Wrong struct packing");

// Restore default packing alignment
#pragma pack(pop)



// Return the center of the light (for area lights averaged position,
// for directional light direction)
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const PointLight& light) {
	return light.position;
}
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const SpotLight& light) {
	return light.position;
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const AreaLightTriangle<CURRENT_DEV>& light) {
	return ei::center(ei::Triangle{ light.points[0u], light.points[1u],
									light.points[2u] });
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const AreaLightQuad<CURRENT_DEV>& light) {
	return ei::center(ei::Tetrahedron{ light.points[0u], light.points[1u],
									   light.points[2u], light.points[3u] });
}
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const AreaLightSphere<CURRENT_DEV>& light) {
	return light.position;
}
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const DirectionalLight& light) {
	return light.direction;
}
CUDA_FUNCTION __forceinline__ const ei::Vec3 get_center(const EnvMapLight<CURRENT_DEV>& light) {
	return ei::Vec3{0.0f};
}

// Converts the light's inert radiometric property into flux
Spectrum get_flux(const PointLight& light);
Spectrum get_flux(const SpotLight& light);
Spectrum get_flux(const AreaLightTriangle<Device::CPU>& light);
Spectrum get_flux(const AreaLightQuad<Device::CPU>& light);
Spectrum get_flux(const AreaLightSphere<Device::CPU>& light);
Spectrum get_flux(const DirectionalLight& light, const ei::Vec3& aabbDiag);
inline Spectrum get_flux(const EnvMapLight<Device::CPU>& light) { return light.flux; }


// ************************************************************************* //
// *** LIGHT SOURCE EVALUATION ********************************************* //
// ************************************************************************* //

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

CUDA_FUNCTION __forceinline__ math::EvalValue
evaluate_point(const Spectrum& intensity) {
	return { intensity, 0.0f, AngularPdf{ 1.0f / (4*ei::PI) }, AngularPdf{ 0.0f } };
}

CUDA_FUNCTION __forceinline__ math::EvalValue
evaluate_spot(const scene::Direction& excident,
			  const Spectrum& intensity,
			  const scene::Direction& spotDir,
			  half cosThetaMax, half cosFalloffStart) {
	const float cosOut = dot(spotDir, excident);
	// Early out
	const float cosThetaMaxf = __half2float(cosThetaMax);
	if(cosOut <= cosThetaMaxf) return math::EvalValue{};
	// OK, there will be some contribution
	const float cosFalloffStartf = __half2float(cosFalloffStart);
	const float falloff = get_falloff(cosOut, cosThetaMaxf, cosFalloffStartf);
	return { intensity * falloff, 0.0f,
			 AngularPdf{ math::get_uniform_cone_pdf(cosThetaMaxf) },
			 AngularPdf{ 0.0f } };
}

CUDA_FUNCTION __forceinline__ math::EvalValue
evaluate_area(const scene::Direction& excident, const Spectrum& intensity,
			  const scene::Direction& normal) {
	const float cosOut = dot(normal, excident);
	// Early out (wrong hemisphere)
	if(cosOut <= 0.0f) return math::EvalValue{};
	return { intensity * cosOut, cosOut, AngularPdf{ cosOut / ei::PI },
			 AngularPdf{ 0.0f } };
}

CUDA_FUNCTION __forceinline__ math::EvalValue
evaluate_dir(const Spectrum& intensity, bool isEnvMap, AngularPdf pdf) {
	// Special case: the incindent area PDF is directly projected.
	// To avoid the wrong conversion later we need to do its reversal here.
	return { intensity, isEnvMap ? 1.0f : 0.0f, pdf, AngularPdf{0.0f} };
}

}}} // namespace mufflon::scene::lights
