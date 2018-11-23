#pragma once

#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "export/api.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/types.hpp"
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
 */
struct alignas(16) AreaLightTriangle {
	alignas(16) ei::Vec3 points[3u];
	alignas(16) ei::Vec3 radiance;
};
struct alignas(16) AreaLightQuad {
	ei::Vec3 points[4u];
	alignas(16) ei::Vec3 radiance;
};
struct alignas(16) AreaLightSphere {
	ei::Vec3 position;
	float radius;
	alignas(16) ei::Vec3 radiance;
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
struct EnvMapLight {
	alignas(16) textures::ConstTextureDevHandle_t<dev> texHandle;
	alignas(16) ei::Vec3 flux;
};

// Asserts to make sure the compiler actually followed our orders
static_assert(sizeof(PointLight) == 32 && alignof(PointLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(SpotLight) == 32 && alignof(SpotLight) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightTriangle) == 64 && alignof(AreaLightTriangle) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightQuad) == 64 && alignof(AreaLightQuad) == 16,
			  "Wrong struct packing");
static_assert(sizeof(AreaLightSphere) == 32 && alignof(AreaLightSphere) == 16,
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
CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const AreaLightTriangle& light) {
	return ei::center(ei::Triangle{ light.points[0u], light.points[1u],
									light.points[2u] });
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_center(const AreaLightQuad& light) {
	return ei::center(ei::Tetrahedron{ light.points[0u], light.points[1u],
									   light.points[2u], light.points[3u] });
}
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const AreaLightSphere& light) {
	return light.position;
}
CUDA_FUNCTION __forceinline__ const ei::Vec3& get_center(const DirectionalLight& light) {
	return light.direction;
}

// Converts the light's inert radiometric property into flux
// These are not in a CPP file because their small size makes them
// prime targets for inlining
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const PointLight& light) {
	return light.intensity * 4.f * ei::PI;
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const SpotLight& light) {
	// Flux for the PBRT spotlight version with falloff ((cos(t)-cos(t_w))/(cos(t_f)-cos(t_w)))^4
	float cosFalloff = __half2float(light.cosFalloffStart);
	float cosWidth = __half2float(light.cosThetaMax);
	return light.intensity * (2.f * ei::PI * (1.f - cosFalloff) + (cosFalloff - cosWidth) / 5.f);
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const AreaLightTriangle& light) {
	return light.radiance * ei::surface(ei::Triangle{
		light.points[0u], light.points[1u], light.points[2u]
										});
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const AreaLightQuad& light) {
	return light.radiance * ei::surface(ei::Tetrahedron{
		light.points[0u], light.points[1u], light.points[2u], light.points[3u]
										});
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const AreaLightSphere& light) {
	return light.radiance * ei::surface(ei::Sphere{ light.position, light.radius });
}
CUDA_FUNCTION __forceinline__ ei::Vec3 get_flux(const DirectionalLight& light,
												const ei::Vec3& aabbDiag) {
	mAssert(aabbDiag.x > 0 && aabbDiag.y > 0 && aabbDiag.z > 0);
	float surface = aabbDiag.y*aabbDiag.z*std::abs(light.direction.x)
		+ aabbDiag.x*aabbDiag.z*std::abs(light.direction.y)
		+ aabbDiag.x*aabbDiag.y*std::abs(light.direction.z);
	return light.radiance * surface;
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



#ifndef __CUDACC__
// Kind of code duplication, but for type-safety use this when constructing a light tree
using PositionalLights = std::variant<PointLight, SpotLight, AreaLightTriangle,
	AreaLightQuad, AreaLightSphere>;
using Light = std::variant<PointLight, SpotLight, AreaLightTriangle,
	AreaLightQuad, AreaLightSphere, DirectionalLight, EnvMapLight<Device::CPU>>;


// Code to detect if a light is of a given type
namespace lights_detail {
// Utility to check if a type is part of a list of types
template < class T, class H, class... Args >
struct IsAnyOf {
	static constexpr bool value = std::is_same_v<T, H> || IsAnyOf<T, Args...>::value;
};
template < class T, class H >
struct IsAnyOf<T, H> {
	static constexpr bool value = std::is_same_v<T, H>;
};
} // namespace lights_detail

template < class T >
inline constexpr bool is_positional_light_type() {
	return lights_detail::IsAnyOf<T, PointLight, SpotLight, AreaLightSphere,
		AreaLightTriangle, AreaLightQuad>::value;
}

template < class T >
inline constexpr bool is_envmap_light_type() {
	return lights_detail::IsAnyOf<T, EnvMapLight<Device::CPU>,
		EnvMapLight<Device::CUDA>, EnvMapLight<Device::OPENGL>>::value;
}

template < class T >
inline constexpr bool is_light_type() {
	return is_positional_light_type<T>() || is_envmap_light_type<T>()
		|| std::is_same_v<T, DirectionalLight>;
}

// Gets the light type as an enum value
template < class LT >
inline LightType get_light_type(const LT& light) {
	static_assert(is_light_type<LT>() || std::is_same_v<LT, PositionalLights>,
				  "Must be light");

	auto posLightType = [](const auto& posLight) constexpr -> LightType {
		using Type = std::decay_t<decltype(posLight)>;
		if constexpr(std::is_same_v<Type, PointLight>)
			return LightType::POINT_LIGHT;
		else if constexpr(std::is_same_v<Type, SpotLight>)
			return LightType::SPOT_LIGHT;
		else if constexpr(std::is_same_v<Type, AreaLightTriangle>)
			return LightType::AREA_LIGHT_TRIANGLE;
		else if constexpr(std::is_same_v<Type, AreaLightQuad>)
			return LightType::AREA_LIGHT_QUAD;
		else if constexpr(std::is_same_v<Type, AreaLightSphere>)
			return LightType::AREA_LIGHT_SPHERE;
		else
			return LightType::NUM_LIGHTS;
	};

	(void)light;
	if constexpr(std::is_same_v<LT, PositionalLights>)
		return std::visit(posLightType, light);
	else if constexpr(is_positional_light_type<LT>())
		return posLightType(light);
	else if constexpr(is_envmap_light_type<LT>())
		return LightType::ENVMAP_LIGHT;
	else if constexpr(std::is_same_v<LT, DirectionalLight>)
		return LightType::DIRECTIONAL_LIGHT;
	else
		return LightType::NUM_LIGHTS;
}

inline ei::Vec3 get_flux(const PositionalLights& light) {
	return std::visit([](const auto& posLight) {
		return get_flux(posLight);
	}, light);
}

template < class LT >
inline ei::Vec3 get_flux(const LT& light, const ei::Vec3& aabbDiag) {
	(void)aabbDiag;
	if constexpr(std::is_same_v<LT, PositionalLights>)
		return std::visit([](const auto& posLight) {
		return get_flux(posLight);
	}, light);
	else if constexpr(is_envmap_light_type<LT>())
		return light.flux;
	else if constexpr(std::is_same_v<LT, DirectionalLight>)
		return get_flux(light, aabbDiag);
	else
		return get_flux(light);
}

#endif // __CUDACC__

}}} // namespace mufflon::scene::lights