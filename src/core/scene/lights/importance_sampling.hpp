#pragma once

#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace lights {

namespace summed_details {

/**
 * Binary search on texel range (either row- or columnwise).
 * Returns the first element that is >= val.
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
		const int step = count / 2;
		const int curr = left + step;
		Pixel texel = row ? Pixel{ constCoord, curr } : Pixel{ curr, constCoord };
		const float x = textures::read(texture, texel).x;
		if(x < val) {
			left = curr + 1;
			count -= step + 1;
		} else {
			count = step;
		}
	}
	return left;
}


} // namespace summed_details

#ifndef __CUDA_ARCH__
inline __host__ void create_summed_area_table(TextureHandle envmap, TextureHandle sums) {
	auto envmapTex = envmap->acquire_const<Device::CPU>();
	auto sumSurf = sums->aquire<Device::CPU>();
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

// The resuult of importance sampling an envmap
struct EnvmapSampleResult {
	Pixel texel;
	ei::Vec2 uv;
	float pdf;
};

/**
 * Importance-samples a texel from a spherical envmap.
 * Expects both the envmap and a summed area table of its luminance.
 * Requires two uniform random numbers in [0, 1].
 * If the envmap is a cube map, the UV.x coordinate will be in [0, 6] instead.
 * Similarly, texel.x will range from 0 to 6*envmap width.
 */
CUDA_FUNCTION EnvmapSampleResult importance_sample_envmap(const EnvMapLight<CURRENT_DEV>& envLight,
														  float u0, float u1) {
	using namespace summed_details;

	// First decide on the row
	const Pixel texSize = textures::get_texture_size(envLight.texHandle);
	const Pixel bottomRight = texSize - 1;
	const float highestRowwise = textures::read(envLight.summedAreaTable, bottomRight).x;

	// Find the row via binary search
	const float x = highestRowwise * u0;
	const int row = lower_bound(envLight.summedAreaTable, x, bottomRight.y, bottomRight.x, true);
	// Perform inverse linear interpolation
	const float vr0 = row == 0 ? 0.0f : textures::read(envLight.summedAreaTable, Pixel{ bottomRight.x, row-1 }).x;
	const float vr1 = textures::read(envLight.summedAreaTable, Pixel{ bottomRight.x, row }).x;
	const float rowPdf = (vr1 - vr0) * texSize.y / highestRowwise;
	const float rowVal = (row + (x - vr0) / (vr1 - vr0)) / static_cast<float>(texSize.y);

	// Decide on a column
	// Adjust the value due to the summing at the end (-vr0)
	float highestColumnwise = vr1 - vr0;

	// Find the column via binary search
	const float y = highestColumnwise * u1;
	const int column = lower_bound(envLight.summedAreaTable, y, bottomRight.x, row, false);
	// Perform inverse linear interpolation
	const float vc0 = column == 0 ? 0.f : textures::read(envLight.summedAreaTable, Pixel{ column - 1, row }).x;
	const float vc1 = textures::read(envLight.summedAreaTable, Pixel{ column, row }).x;
	const float columnPdf = (vc1 - vc0) * texSize.x / highestColumnwise;
	const float columnVal = (column + (y - vc0) / (vc1 - vc0)) / static_cast<float>(texSize.x);

	return EnvmapSampleResult{ Pixel{row, column}, ei::Vec2{ rowVal, columnVal }, rowPdf * columnPdf };
}

}}} // namespace mufflon::scene::lights