#pragma once

#include "export/api.hpp"
#include "texture.hpp"
#include "cputexture.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace textures {

// Read a CPU or CUDA texture
inline __host__ ei::Vec4 read(ConstTextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer = 0) {
	return texture->read(texel, layer);
}
inline __host__ ei::Vec4 read(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer = 0) {
	return texture->read(texel, layer);
}

#ifdef __CUDACC__
inline __device__ ei::Vec4 read(ConstTextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer = 0) {
	int idx = texel.x + (texel.y + layer * texture.height) * texture.width;
	return ei::details::hard_cast<ei::Vec4>( tex1Dfetch<float4>(texture.handle, idx) );
}
inline __device__ ei::Vec4 read(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer = 0) {
	return ei::details::hard_cast<ei::Vec4>(
		surf2DLayeredread<float4>(texture.handle, texel.x, texel.y, layer));
}
#endif

// Write a CPU or CUDA texture
inline __host__ void write(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, const ei::Vec4& value) {
	texture->write(value, texel, 0);
}
inline __host__ void write(TextureDevHandle_t<Device::CPU> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
	texture->write(value, texel, layer);
}

#ifdef __CUDACC__
inline __device__ void write(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, const ei::Vec4& value) {
	float4 v = ei::details::hard_cast<float4>(value);
	surf2Dwrite<float4>(v, texture.handle, texel.x, texel.y);
}
inline __device__ void write(TextureDevHandle_t<Device::CUDA> texture, const Pixel& texel, int layer, const ei::Vec4& value) {
	float4 v = ei::details::hard_cast<float4>(value);
	surf2DLayeredwrite<float4>(v, texture.handle, texel.x, texel.y, layer);
}
#endif


// Sample a CPU or CUDA texture
inline __host__ ei::Vec4 sample(ConstTextureDevHandle_t<Device::CPU> texture, const UvCoordinate& uv) {
	return texture->sample(uv);
}

#ifdef __CUDACC__
inline __device__ ei::Vec4 sample(ConstTextureDevHandle_t<Device::CUDA> texture, const UvCoordinate& uv) {
	auto texel = tex2D<float4>(texture.handle, uv.x, uv.y);
	return ei::details::hard_cast<ei::Vec4>( texel );
}
#endif

}}} // namespace mufflon::scene::textures