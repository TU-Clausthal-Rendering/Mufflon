#pragma once

#include "ei/vector.hpp"
#include "core/scene/residency.hpp"
#include "core/scene/types.hpp"
#include "core/scene/textures/texture.hpp"

namespace mufflon { namespace scene { namespace lights {

enum class LightType : u32 {
	POINT_LIGHT,
	SPOT_LIGHT,
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
	ei::Vec3 position;
	float padding1;
	ei::Vec3 intensity;
	float padding2;
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
 * Directional light. Doesn't reduce bitness of encoded direction.
 */
struct alignas(16) DirectionalLight {
	ei::Vec3 position;
	ei::Vec3 intensity;
	ei::Vec2 direction;
};

/**
 * Environment-map light.
 */
template < Device dev >
struct alignas(16) EnvMapLight {
	textures::DeviceTextureHandle<dev> envMap;
};

// Asserts to make sure the compiler actually followed our orders
static_assert(sizeof(PointLight) == 32 && alignof(PointLight) == 16);
static_assert(sizeof(SpotLight) == 32 && alignof(SpotLight) == 16);
static_assert(sizeof(DirectionalLight) == 32 && alignof(DirectionalLight) == 16);
static_assert(sizeof(EnvMapLight<Device::CPU>) == 16 && alignof(EnvMapLight<Device::CPU>) == 16);
static_assert(sizeof(EnvMapLight<Device::CUDA>) == 16 && alignof(EnvMapLight<Device::CUDA>) == 16);

// Restore default packing alignment
#pragma pack(pop)

}}} // namespace mufflon::scene::lights