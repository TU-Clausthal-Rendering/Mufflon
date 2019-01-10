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
	ei::Vec3 position {0.0f};
	materials::MediumHandle mediumIndex {u16(~0u)};
	alignas(16) Spectrum intensity {1.0f};
};

/**
 * A spot light, following PBRT's description.
 * To save storage space, the direction is encoded in a single u32.
 * Additionally, the cosine terms for opening angle and falloff
 * are encoded 
 */
struct alignas(16) SpotLight {
    ei::Vec3 position {0.0f};
    half cosThetaMax {0.5f};
    half cosFalloffStart {0.7f};
    ei::Vec3 direction {0.0f, -1.0f, 0.0f};
	materials::MediumHandle mediumIndex {u16(~0u)};
    alignas(16) ei::Vec3 intensity {1.0f};
};

/**
 * Area lights. One for every type of geometric primitive.
 * TODO: use an indirection to the primitive -> 16 Byte per light instead of up to 96.
 *		The cost of the indirection will be a more expensive fetch.
 * TODO: also test uv-compression (shared exponent or fixed point?)
 */
struct alignas(16) AreaLightTriangleDesc {
	alignas(16) ei::Vec3 points[3u];		// 36 bytes
	u32 scale;								// A spectrum scale for the radiance texture in RGB9E5
	alignas(8) UvCoordinate uv[3u];			// 24 bytes
	alignas(8) TextureHandle radianceTex;	// 8 bytes
};
struct alignas(16) AreaLightQuadDesc {
	ei::Vec3 points[4u];					// 48 bytes
	alignas(8) UvCoordinate uv[4u];			// 32 bytes
	alignas(8) TextureHandle radianceTex;	// 8 byes
	u32 scale;								// A spectrum scale for the radiance texture in RGB9E5
};
struct alignas(16) AreaLightSphereDesc {
	ei::Vec3 position;						// 12 bytes
	float radius;							// 4 bytes
	alignas(8) TextureHandle radianceTex;	// 8 bytes
	u32 scale;								// A spectrum scale for the radiance texture in RGB9E5
};

template < Device dev >
struct alignas(16) AreaLightTriangle {
	alignas(16) ei::Vec3 points[3u];
	u32 scale;								// A spectrum scale for the radiance texture in RGB9E5
	alignas(8) UvCoordinate uv[3u];
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;

	AreaLightTriangle& operator=(const AreaLightTriangleDesc& rhs) {
		for(int i = 0; i < 3; ++i) {
			points[i] = rhs.points[i];
			uv[i] = rhs.uv[i];
		}
		scale = rhs.scale;
		radianceTex = rhs.radianceTex->acquire_const<dev>();
		return *this;
	}
};
template < Device dev >
struct alignas(16) AreaLightQuad {
	ei::Vec3 points[4u];
	alignas(8) UvCoordinate uv[4u];
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;
	// The forced alignment makes sure the two types CPU/GPU have the same size.
	// Problem: radianceTex is 16 byte on GPU but only 8 on CPU.
	alignas(16) u32 scale;			// A spectrum scale for the radiance texture in RGB9E5

	AreaLightQuad& operator=(const AreaLightQuadDesc& rhs) {
		for(int i = 0; i < 4; ++i) {
			points[i] = rhs.points[i];
			uv[i] = rhs.uv[i];
		}
		scale = rhs.scale;
		radianceTex = rhs.radianceTex->acquire_const<dev>();
		return *this;
	}
};
template < Device dev >
struct alignas(16) AreaLightSphere {
	ei::Vec3 position;
	float radius;
	alignas(8) textures::ConstTextureDevHandle_t<dev> radianceTex;
	alignas(16) u32 scale;			// A spectrum scale for the radiance texture in RGB9E5

	AreaLightSphere& operator=(const AreaLightSphereDesc& rhs) {
		position = rhs.position;
		radius = rhs.radius;
		scale = rhs.scale;
		radianceTex = rhs.radianceTex->acquire_const<dev>();
		return *this;
	}
};

/**
 * Directional light. Doesn't have a position.
 */
struct alignas(16) DirectionalLight {
	alignas(16) ei::Vec3 direction {0.0f, -1.0f, 0.0f};	// Direction in which the light travels (incident on surfaces)
	alignas(16) ei::Vec3 radiance {1.0f};
};

/**
 * Environment light.
 */
enum class BackgroundType {
	COLORED,
	ENVMAP
};

template < Device dev >
struct alignas(16) BackgroundDesc {
	textures::ConstTextureDevHandle_t<dev> envmap;
	textures::ConstTextureDevHandle_t<dev> summedAreaTable;
	BackgroundType type;
	Spectrum color;				// Color for uniform backgrounds OR scale in case of envLights
	Spectrum flux;
};

// Asserts to make sure the compiler actually followed our orders
static_assert(sizeof(PointLight) == 32 && alignof(PointLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(SpotLight) == 48 && alignof(SpotLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightTriangle<Device::CPU>) == 80 && alignof(AreaLightTriangle<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightTriangle<Device::CUDA>) == 80 && alignof(AreaLightTriangle<Device::CUDA>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightQuad<Device::CPU>) == 112 && alignof(AreaLightQuad<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightQuad<Device::CUDA>) == 112 && alignof(AreaLightQuad<Device::CUDA>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightSphere<Device::CPU>) == 48 && alignof(AreaLightSphere<Device::CPU>) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightSphere<Device::CUDA>) == 48 && alignof(AreaLightSphere<Device::CUDA>) == 16,
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
CUDA_FUNCTION __forceinline__ const ei::Vec3 get_center(const BackgroundDesc<CURRENT_DEV>& light) {
	return ei::Vec3{0.0f};
}

// Converts the light's inert radiometric property into flux
Spectrum get_flux(const PointLight& light);
Spectrum get_flux(const SpotLight& light);
Spectrum get_flux(const AreaLightTriangle<Device::CPU>& light);
Spectrum get_flux(const AreaLightQuad<Device::CPU>& light);
Spectrum get_flux(const AreaLightSphere<Device::CPU>& light);
Spectrum get_flux(const DirectionalLight& light, const ei::Vec3& aabbDiag);


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
