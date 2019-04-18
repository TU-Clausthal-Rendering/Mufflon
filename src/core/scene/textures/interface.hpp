#pragma once

#include "core/export/api.h"
#include "texture.hpp"
#include "cputexture.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

// Read a CPU or CUDA texture
CUDA_FUNCTION ei::Vec4 read(ConstTextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
#ifndef __CUDA_ARCH__
	return texture->read(texel, layer);
#else
	// CUDA textures cannot switch linear filtering on/off, so we assume always-on and normalized coordinates.
	// This means that 1. we need to add 0.5 to the coordinate to get to the center of the texel
	// and 2. we need to normalize the coordinates into range [0, 1]
	return ei::details::hard_cast<ei::Vec4>(tex2DLayered<float4>(texture.handle, (texel.x + 0.5f) / static_cast<float>(texture.width),
																 (texel.y + 0.5f) / static_cast<float>(texture.height), layer));
#endif
}

CUDA_FUNCTION ei::Vec4 read(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
#ifndef __CUDA_ARCH__
	return texture->read(texel, layer);
#else
	// Unlike textures, surfaces are NEVER filtered and do not offer normalized coordinates either.
	// Additionally, surface indices are byte-based (in x-direction, anyway) and thus need to
	// be multiplied with the texel size
	return ei::details::hard_cast<ei::Vec4>(
		surf2DLayeredread<float4>(texture.handle, texel.x * PIXEL_SIZE(texture.format),
								  texel.y, layer));
#endif
}


// Write a CPU or CUDA texture
CUDA_FUNCTION void write(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, const ei::Vec4& value) {
#ifndef __CUDA_ARCH__
	texture->write(value, texel, 0);
#else
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, 0);
#endif
}
CUDA_FUNCTION void write(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
#ifndef __CUDA_ARCH__
	texture->write(value, texel, layer);
#else
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, layer);
#endif
}


// Sample a CPU or CUDA texture
CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<CURRENT_DEV> texture, const UvCoordinate& uv, int layer = 0u) {
#ifndef __CUDA_ARCH__
	return texture->sample(uv, layer);
#else
	// TODO: layer
	// UV coordinates need to be scaled, so that we 
	auto texel = tex2DLayered<float4>(texture.handle, uv.x, uv.y, layer);
	return ei::details::hard_cast<ei::Vec4>(texel);
#endif
}



// Compute the UV coordinate in [0,1]² and the layer given a point on the cube
CUDA_FUNCTION UvCoordinate cubemap_surface_to_uv(const Point& cubePos, int& layer) {
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
CUDA_FUNCTION Point cubemap_uv_to_surface(UvCoordinate uv, int layer) {
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

// Samples an environment map and returns the uv-coordinate too
CUDA_FUNCTION ei::Vec4 sample(ConstTextureDevHandle_t<CURRENT_DEV> envmap, const ei::Vec3& direction, UvCoordinate& uvOut) {
#ifndef __CUDA_ARCH__
	int layers = envmap->get_num_layers();
#else // __CUDA_ARCH__
	int layers = envmap.depth;
#endif // __CUDA_ARCH__
	if(layers == 6) {
		// Cubemap
		// Find out which face by elongating the direction
		ei::Vec3 projDir = direction / ei::max(ei::abs(direction));
		projDir.z = -projDir.z;
		// Set the layer and UV coordinates
		int layer;
		uvOut = cubemap_surface_to_uv(projDir, layer);
		return sample(envmap, uvOut, layer);
	} else {
		// Spherical map
		// Convert the direction into UVs (convention: phi ~ u, theta ~ v)
		float v = acos(direction.y) / ei::PI;
		const float u = atan2(direction.z, direction.x) / (ei::PI * 2.f);
		// Clamp (no wrapping in v direction)
		const Pixel texSize = textures::get_texture_size(envmap);
		v = ei::min(v, (texSize.y - 0.5f) / texSize.y);
		uvOut = UvCoordinate{ u, 1.f-v };
		return sample(envmap, uvOut);
	}
}

}}} // namespace mufflon::scene::textures