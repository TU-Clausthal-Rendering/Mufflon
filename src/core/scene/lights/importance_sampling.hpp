#pragma once

#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace lights {

namespace summed_details {

// Returns the size of a texture based on its handle
template < Device dev >
CUDA_FUNCTION __forceinline__ Pixel get_texture_size(const textures::ConstTextureDevHandle_t<dev>& texture) noexcept;
template <>
inline __host__ __forceinline__ Pixel get_texture_size<Device::CPU>(const textures::ConstTextureDevHandle_t<Device::CPU>& texture) noexcept {
#ifndef __CUDA_ARCH__
	return { texture->get_width(), texture->get_height() };
#else // __CUDA_ARCH__
	return Pixel{};
#endif // __CUDA_ARCH__
}
template <>
CUDA_FUNCTION __forceinline__ Pixel get_texture_size<Device::CUDA>(const textures::ConstTextureDevHandle_t<Device::CUDA>& texture) noexcept {
	return { texture.width, texture.height };
}

/**
	 * Binary search on texel range (either row- or columnwise).
	 * val: Value to compare against (lower bound)
	 * max: Upper limit for the row or column (last valid value)
	 * constCoord: value for the non-variable coordinate
	 * row: find over rows or columns
	 */
CUDA_FUNCTION __forceinline__ int lower_bound(const textures::ConstTextureDevHandle_t<CURRENT_DEV>& texture,
											  const float val, const int max,
											  const int constCoord, const bool row) {

	int left = 0u;
	int count = max + 1;
	while(count > 0) {
		const int curr = count / 2;
		Pixel texel = row ? Pixel{ constCoord, curr } : Pixel{ curr, constCoord };
		const float x = textures::read(texture, texel).x;
		if(val < x) {
			left = curr + 1;
			count -= left;
		} else {
			count = curr;
		}
	}
	return left;
}


} // namespace summed_details

#ifndef __CUDA_ARCH__
inline __host__ void create_summed_area_table(TextureHandle envmap, TextureHandle sums) {
	auto envmapTex = *envmap->aquireConst<Device::CPU>();
	auto sumSurf = *envmap->aquire<Device::CPU>();
	// Conversion to luminance
	constexpr ei::Vec4 lumWeight{ 0.212671f, 0.715160f, 0.072169f, 0.f };
	const int width = static_cast<int>(envmap->get_width());
	const int height = static_cast<int>(envmap->get_height());
	const int layers = static_cast<int>(envmap->get_num_layers());

	if(layers == 6u) {
		// Cubemap
		// Code taken from Johannes' renderer
		// Layers are encoded as continuous in the rows
		float accumLumY = 0.f;
		for(int y = 0; y < height; ++y) {
			float accumLumX = 0.f;
			for(int l = 0; l < layers; ++l) {
				for(int x = 0; x < width; ++x) {
					const float luminance = ei::dot(lumWeight, textures::read(envmapTex, Pixel{ x, y }, l));
					accumLumX += luminance;
					textures::write(sumSurf, Pixel{ x + width * l, y }, ei::Vec4{ accumLumX, 0.f, 0.f, 0.f });
				}
			}
			// In the last texel of a row we store the accumulated PDF for the columns
			accumLumY += accumLumX;
			textures::write(sumSurf, Pixel{ 6 * width - 1, y }, ei::Vec4{ accumLumY, 0.f, 0.f, 0.f });
		}
	} else {
		// Polarmap
		// Code taken from PBRT
		float accumLumY = 0.f;
		for(int y = 0; y < height; ++y) {
			const float sinTheta = sinf(ei::PI * static_cast<float>(y + 0.5f) / static_cast<float>(height));
			float accumLumX = 0.f;
			for(int x = 0; x < width; ++x) {
				const Pixel texel{ x, y };
				const float luminance = ei::dot(lumWeight, textures::read(envmapTex, texel));
				accumLumX += luminance * sinTheta;
				textures::write(sumSurf, texel, ei::Vec4{ accumLumX, 0.f, 0.f, 0.f });
			}

			// In the last texel of a row we store the accumulated PDF for the columns
			accumLumY += accumLumX;
			textures::write(sumSurf, Pixel{ width - 1, y }, ei::Vec4{ accumLumY, 0.f, 0.f, 0.f });
		}
	}
}
#endif // __CUDA_ARCH__

/**
 * Importance-samples a texel from a spherical envmap.
 * Expects both the envmap and a summed area table of its luminance.
 * Requires two uniform random numbers in [0, 1].
 */
CUDA_FUNCTION Pixel importance_sample_envmap(const textures::ConstTextureDevHandle_t<CURRENT_DEV>& envmap,
											 const textures::ConstTextureDevHandle_t<CURRENT_DEV>& summedAreaTable,
											 float u0, float u1) {
	using namespace summed_details;

		// First decide on the row
	const Pixel texSize = get_texture_size<CURRENT_DEV>(envmap);
	const Pixel bottomRight = texSize - 1;
	const float highestRowwise = textures::read(summedAreaTable, bottomRight).x;

	// Find the row via binary search
	const float x = highestRowwise * u0;
	const int row = lower_bound(summedAreaTable, x, bottomRight.y, bottomRight.x, true);

	// Decide on a column
	float highestColumnwise = textures::read(summedAreaTable, Pixel{ bottomRight.x, row }).x;
	// Adjust the value due to the summing at the end
	if(row > 0)
		highestColumnwise -= textures::read(summedAreaTable, Pixel{ bottomRight.x, row - 1 }).x;

	// Find the column via binary search
	const float y = highestColumnwise * u1;
	const int column = lower_bound(summedAreaTable, y, bottomRight.x, row, false);
	return Pixel{ row, column };
}

}}} // namespace mufflon::scene::lights