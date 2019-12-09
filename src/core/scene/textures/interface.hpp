#pragma once

#include "core/export/core_api.h"
#include "texture.hpp"
#include "cputexture.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

// Read a CPU or CUDA texture
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION ei::Vec4 read(ConstTextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer = 0, int level = 0) {
	return texture->read(texel, layer, level);
}
inline CUDA_FUNCTION ei::Vec4 read(ConstTextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer = 0, int level = 0) {
#ifdef __CUDA_ARCH__
	// CUDA textures cannot switch linear filtering on/off, so we assume always-on and normalized coordinates.
	// This means that 1. we need to add 0.5 to the coordinate to get to the center of the texel
	// and 2. we need to normalize the coordinates into range [0, 1]
	return ei::details::hard_cast<ei::Vec4>(tex2DLayeredLod<float4>(texture.handle, (texel.x + 0.5f) / static_cast<float>(texture.width),
		(texel.y + 0.5f) / static_cast<float>(texture.height), layer, level));
#else // __CUDA_ARCH__
	(void)texture;
	(void)texel;
	(void)layer;
	(void)level;
	return {};
#endif // __CUDA_ARCH__
}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION ei::Vec4 read(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer = 0) {
	// We only support reading from the top-level mipmap for read/write textures since CUDA surfaces can only bind to a single mipmap
	return texture->read(texel, layer);
}
inline CUDA_FUNCTION ei::Vec4 read(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer = 0) {
#ifdef __CUDA_ARCH__
	// Unlike textures, surfaces are NEVER filtered and do not offer normalized coordinates either.
	// Additionally, surface indices are byte-based (in x-direction, anyway) and thus need to
	// be multiplied with the texel size
	return ei::details::hard_cast<ei::Vec4>(
		surf2DLayeredread<float4>(texture.handle, texel.x * PIXEL_SIZE(texture.format),
								  texel.y, layer));
#else // __CUDA_ARCH__
	(void)texture;
	(void)texel;
	(void)layer;
	return {};
#endif // __CUDA_ARCH__
}


// Write a CPU or CUDA texture
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION void write(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, const ei::Vec4& value) {
	texture->write(value, texel, 0);
}
inline CUDA_FUNCTION void write(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, const ei::Vec4& value) {
#ifdef __CUDA_ARCH__
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, 0);
#endif // __CUDA_ARCH__
	(void)texture;
	(void)texel;
	(void)value;
}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION void write(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
	// We only support writing to the top-level mipmap since CUDA surfaces can only bind to a single mipmap
	texture->write(value, texel, layer);
}
inline CUDA_FUNCTION void write(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
#ifdef __CUDA_ARCH__
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, layer);
#endif // __CUDA_ARCH__
	(void)texture;
	(void)texel;
	(void)layer;
	(void)value;
}


// Sample a CPU or CUDA texture
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<Device::CPU> texture, const UvCoordinate& uv, int layer = 0u, float level = 0.f) {
	return texture->sample(uv, layer, level);
}

// Sample a CPU or CUDA texture
inline CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<Device::CUDA> texture, const UvCoordinate& uv, int layer = 0u, float level = 0.f) {
	// TODO: layer
	// UV coordinates need to be scaled, so that we 
#ifdef __CUDA_ARCH__
	auto texel = tex2DLayeredLod<float4>(texture.handle, uv.x, uv.y, layer, level);
	return ei::details::hard_cast<ei::Vec4>(texel);
#else // __CUDA_ARCH__
	(void)texture;
	(void)uv;
	(void)layer;
	(void)level;
	return {};
#endif // __CUDA_ARCH__
}



// Compute the UV coordinate in [0,1]� and the layer given a point on the cube
inline CUDA_FUNCTION UvCoordinate cubemap_surface_to_uv(const Point& cubePos, int& layer) {
	// Set the layer and UV coordinates
	float u, v;
	if(cubePos.x == 1.f) {
		layer = 0u;
		u = -cubePos.z;
		v = cubePos.y;
	} else if(cubePos.x == -1.f) {
		layer = 1u;
		u = cubePos.z;
		v = cubePos.y;
	} else if(cubePos.y == 1.f) {
		layer = 2u;
		u = cubePos.x;
		v = -cubePos.z;
	} else if(cubePos.y == -1.f) {
		layer = 3u;
		u = cubePos.x;
		v = cubePos.z;
	} else if(cubePos.z == 1.f) {
		layer = 4u;
		u = cubePos.x;
		v = cubePos.y;
	} else {
		layer = 5u;
		u = -cubePos.x;
		v = cubePos.y;
	}
	// Normalize the UV coordinates into [0, 1]
	return { (u + 1.f) / 2.f, (v + 1.f) / 2.f };
}

// Compute the postion on the unit cube given a layer index and the local uv coordinates.
inline CUDA_FUNCTION Point cubemap_uv_to_surface(UvCoordinate uv, int layer) {
	// Turn the texel coordinates into UVs and remap from [0, 1] to [-1, 1]
	uv = uv * 2.0f - 1.0f;
	switch(layer) {
		case 0: return Point{ 1.f, uv.y, -uv.x };
		case 1: return Point{ -1.f, uv.y, uv.x };
		case 2: return Point{ uv.x, 1.f, -uv.y };
		case 3: return Point{ uv.x, -1.f, uv.y };
		case 4: return Point{ uv.x, uv.y, 1.f };
		case 5:
		default:
			return Point{ -uv.x, uv.y, -1.f };
	}
}

// Get the uv coordinate of a direction by using polar coordinates.
CUDA_FUNCTION __forceinline__ UvCoordinate direction_to_uv(const Direction& direction) {
	// Convert the direction into UVs (convention: phi ~ u, theta ~ v)
	float v = acos(direction.y) / ei::PI;
	const float u = atan2(direction.z, direction.x) / (ei::PI * 2.0f);
	return UvCoordinate{ u, 1.0f-v };
}

template < Device dev >
inline CUDA_FUNCTION ei::Vec4 sample_cubemap(ConstTextureDevHandle_t<dev> cubemap, const ei::Vec3& direction, UvCoordinate& uvOut) {
	// Find out which face by elongating the direction
	ei::Vec3 projDir = direction / ei::max(ei::abs(direction));
	projDir.z = -projDir.z;
	// Set the layer and UV coordinates
	int layer;
	uvOut = cubemap_surface_to_uv(projDir, layer);
	return sample(cubemap, uvOut, layer);
}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
template < Device dev >
inline CUDA_FUNCTION ei::Vec4 sample_polar_map(ConstTextureDevHandle_t<dev> polarmap, const ei::Vec3& direction, UvCoordinate& uvOut) {
	// Convert the direction into UVs (convention: phi ~ u, theta ~ v)
	uvOut = direction_to_uv(direction);
	// Clamp (no wrapping in v direction)
	const Pixel texSize = textures::get_texture_size(polarmap);
	float halfTexel = 0.5f / texSize.y;
	uvOut.v = ei::clamp(uvOut.v, halfTexel, 1.0f - halfTexel);
	return sample(polarmap, uvOut);
}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif // _CUDACC__
inline CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<Device::CPU> envmap, const ei::Vec3& direction, UvCoordinate& uvOut) {
	int layers = envmap->get_num_layers();
	if(layers == 6)
		return sample_cubemap<Device::CPU>(envmap, direction, uvOut);
	else
		return sample_polar_map<Device::CPU>(envmap, direction, uvOut);
}
inline CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<Device::CUDA> envmap, const ei::Vec3& direction, UvCoordinate& uvOut) {
	int layers = envmap.depth;
	if(layers == 6)
		return sample_cubemap<Device::CUDA>(envmap, direction, uvOut);
	else
		return sample_polar_map<Device::CUDA>(envmap, direction, uvOut);
}


}}} // namespace mufflon::scene::textures
