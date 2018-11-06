#pragma once

#include "ei/vector.hpp"
#include "core/scene/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"
#include <variant>

namespace mufflon { namespace scene { namespace lights {

enum class LightType : u32 {
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
 * To save storage space, the direction is encoded in a single float.
 * Additionally, the cosine terms for opening angle and falloff
 * are encoded 
 */
struct alignas(16) SpotLight {
	ei::Vec3 position;
	u32 direction;
	ei::Vec3 intensity;
	u32 angles;
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
	alignas(16) textures::ConstDeviceTextureHandle<dev> texHandle;
	alignas(16) ei::Vec3 flux;
};

// Encodes all possibilities for light sources which have a spatial location
union PositionalLight {
	PointLight point;
	SpotLight spot;
	AreaLightTriangle areaTri;
	AreaLightQuad areaQuad;
	AreaLightSphere areaSphere;
};

// Kind of code duplication, but for type-safety use this when constructing a light tree
using PositionalLights = std::variant<PointLight, SpotLight, AreaLightTriangle,
	AreaLightQuad, AreaLightSphere>;

// Asserts to make sure the compiler actually followed our orders
static_assert(sizeof(PointLight) == 32 && alignof(PointLight) == 16);
static_assert(sizeof(SpotLight) == 32 && alignof(SpotLight) == 16);
static_assert(sizeof(AreaLightTriangle) == 64 && alignof(AreaLightTriangle) == 16);
static_assert(sizeof(AreaLightQuad) == 64 && alignof(AreaLightQuad) == 16);
static_assert(sizeof(AreaLightSphere) == 32 && alignof(AreaLightSphere) == 16);
static_assert(sizeof(DirectionalLight) == 32 && alignof(DirectionalLight) == 16);

// Restore default packing alignment
#pragma pack(pop)

}}} // namespace mufflon::scene::lights