#pragma once

#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/texture.hpp"
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



#ifndef __CUDACC__
// Kind of code duplication, but for type-safety use this when constructing a light tree
struct PositionalLights {
	std::variant<PointLight, SpotLight, AreaLightTriangleDesc,
				 AreaLightQuadDesc, AreaLightSphereDesc> light;
	PrimitiveHandle primitive { ~0u };
};

// Gets the light type as an enum value
inline LightType get_light_type(const PositionalLights& light) {
	return std::visit([](const auto& posLight) constexpr -> LightType {
		using Type = std::decay_t<decltype(posLight)>;
		if constexpr(std::is_same_v<Type, PointLight>)
			return LightType::POINT_LIGHT;
		else if constexpr(std::is_same_v<Type, SpotLight>)
			return LightType::SPOT_LIGHT;
		else if constexpr(std::is_same_v<Type, AreaLightTriangle<CURRENT_DEV>>)
			return LightType::AREA_LIGHT_TRIANGLE;
		else if constexpr(std::is_same_v<Type, AreaLightQuad<CURRENT_DEV>>)
			return LightType::AREA_LIGHT_QUAD;
		else if constexpr(std::is_same_v<Type, AreaLightSphere<CURRENT_DEV>>)
			return LightType::AREA_LIGHT_SPHERE;
		else
			return LightType::NUM_LIGHTS;
	}, light.light);
}

#endif // __CUDACC__

}}} // namespace mufflon::scene::lights