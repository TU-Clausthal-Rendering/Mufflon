#pragma once

#include "core/export/api.h"
#include "texture.hpp"
#include "cputexture.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

// Read a CPU or CUDA texture
inline __host__ __device__ ei::Vec4 read(ConstTextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
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

inline __host__ __device__ ei::Vec4 read(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
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
inline __host__ __device__ void write(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, const ei::Vec4& value) {
#ifndef __CUDA_ARCH__
	texture->write(value, texel, 0);
#else
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, 0);
#endif
}
inline __host__ __device__ void write(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
#ifndef __CUDA_ARCH__
	texture->write(value, texel, layer);
#else
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x * PIXEL_SIZE(texture.format), texel.y, layer);
#endif
}


// Sample a CPU or CUDA texture
inline __host__ __device__ ei::Vec4 sample(ConstTextureDevHandle_t<CURRENT_DEV> texture, const UvCoordinate& uv, int layer = 0u) {
#ifndef __CUDA_ARCH__
	return texture->sample(uv, layer);
#else
	// TODO: layer
	// UV coordinates need to be scaled, so that we 
	auto texel = tex2DLayered<float4>(texture.handle, uv.x, uv.y, layer);
	return ei::details::hard_cast<ei::Vec4>(texel);
#endif
}

// Samples an environment map
inline __host__ __device__ ei::Vec4 sample(ConstTextureDevHandle_t<CURRENT_DEV> envmap, const ei::Vec3& direction) {
#ifndef __CUDA_ARCH__
	int layers = envmap->get_num_layers();
#else // __CUDA_ARCH__
	int layers = envmap.depth;
#endif // __CUDA_ARCH__
	if(layers == 6) {
		// Cubemap
		// Find out which face by elongating the direction
		ei::Vec3 projDir = direction / ei::max(direction);
		// Set the layer and UV coordinates
		int layer;
		float u, v;
		if(projDir.x == 1.f) {
			layer = 0u;
			u = projDir.y;
			v = -projDir.z;
		} else if(projDir.x == -1.f) {
			layer = 1u;
			u = projDir.y;
			v = projDir.z;
		} else if(projDir.y == 1.f) {
			layer = 2u;
			u = -projDir.z;
			v = projDir.x;
		} else if(projDir.y == -1.f) {
			layer = 3u;
			u = projDir.z;
			v = projDir.x;
		} else if(projDir.z == 1.f) {
			layer = 4u;
			u = projDir.y;
			v = projDir.x;
		} else {
			layer = 5u;
			u = projDir.y;
			v = -projDir.x;
		}
		// Normalize the UV coordinates into [0, 1]
		u = (u + 1.f) / 2.f;
		v = (v + 1.f) / 2.f;
		return sample(envmap, UvCoordinate{ u, v }, layer);
	} else {
		// Spherical map
		// Convert the direction into UVs (convention: phi ~ u, theta ~ v)
		const float u = atan2(direction.y, direction.x) / (ei::PI * 2.f);
		const float v = acos(direction.z) / ei::PI;
		return sample(envmap, UvCoordinate{ u, v });
	}
}

}}} // namespace mufflon::scene::textures