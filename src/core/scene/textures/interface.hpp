#pragma once

#include "export/api.hpp"
#include "texture.hpp"
#include "cputexture.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

// Read a CPU or CUDA texture
inline __host__ __device__ ei::Vec4 read(ConstTextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
#ifndef __CUDA_ARCH__
	return texture->read(texel, layer);
#else
	int idx = texel.x + (texel.y + layer * texture.height) * texture.width;
	return ei::details::hard_cast<ei::Vec4>( tex1Dfetch<float4>(texture.handle, idx) );
#endif
}
inline __host__ __device__ ei::Vec4 read(TextureDevHandle_t<CURRENT_DEV> texture, const Pixel& texel, int layer = 0) {
#ifndef __CUDA_ARCH__
	return texture->read(texel, layer);
#else
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
inline __host__ __device__ ei::Vec4 sample(ConstTextureDevHandle_t<CURRENT_DEV> texture, const UvCoordinate& uv) {
#ifndef __CUDA_ARCH__
	return texture->sample(uv);
#else
	auto texel = tex2D<float4>(texture.handle, uv.x, uv.y);
	return ei::details::hard_cast<ei::Vec4>( texel );
#endif
}

}}} // namespace mufflon::scene::textures