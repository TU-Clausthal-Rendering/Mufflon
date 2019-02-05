#pragma once

#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
#include "core/scene/textures/interface.hpp"
#include "util/flag.hpp"
#include "path_util.hpp"

namespace mufflon { namespace renderer {

struct RenderTargets {
	static constexpr u32 RADIANCE = 0u;
	static constexpr u32 POSITION = 1u;
	static constexpr u32 ALBEDO = 2u;
	static constexpr u32 NORMAL = 3u;
	static constexpr u32 LIGHTNESS = 4u;
};

struct OutputValue : util::Flags<u32> {
	static constexpr u32 RADIANCE = 0x0001;			// Output of radiance (standard output of a renderer)
	static constexpr u32 POSITION = 0x0002;			// Output of positions (customly integrated over paths)
	static constexpr u32 ALBEDO = 0x0004;			// Output of albedo (customly integrated over paths)
	static constexpr u32 NORMAL = 0x0008;			// Output of signed normals (customly integrated over paths)
	static constexpr u32 LIGHTNESS = 0x0010;		// Output of multiplied paths weights (from the custom integration)

	static constexpr u32 RADIANCE_VAR = 0x0100;		// Variance of radiance
	static constexpr u32 POSITION_VAR = 0x0200;
	static constexpr u32 ALBEDO_VAR = 0x0400;
	static constexpr u32 NORMAL_VAR = 0x0800;
	static constexpr u32 LIGHTNESS_VAR = 0x1000;

	static constexpr u32 iterator[5] = {RADIANCE, POSITION, ALBEDO, NORMAL, LIGHTNESS};

	bool is_variance() const { return (mask & 0xff00) != 0; }
};

template < Device dev >
struct RenderBuffer {
	// The following texture handles may contain iteration only or all iterations summed
	// information. The meaning of the variables is only known to the OutputHandler.
	// The renderbuffer only needs to add values to all defined handles.

	scene::textures::TextureDevHandle_t<dev> m_targets[5u] = {};
	ei::IVec2 m_resolution;

	/*
	 * Handle contribution of connection and merge events
	 * value: The radiance estimate from the event. This can be the BxDF (merge) or
	 *		BxDF * BxDF / distSq for connections.
	 */
	__host__ __device__ void contribute(Pixel pixel,
										const Throughput & viewThroughput,
										const Throughput & lightThroughput,
										float cosines, const ei::Vec3& value
	) {
		using namespace scene::textures;
		if(is_valid(m_targets[RenderTargets::RADIANCE])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::RADIANCE], pixel);
			ei::Vec3 newVal = viewThroughput.weight * lightThroughput.weight * value * cosines;
			mAssert(!isnan(newVal.x) && !isnan(newVal.y) && !isnan(newVal.z));
			write(m_targets[RenderTargets::RADIANCE], pixel, prev+ei::Vec4{newVal, 0.0f});
		}
		if(is_valid(m_targets[RenderTargets::LIGHTNESS])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::LIGHTNESS], pixel);
			float newVal = viewThroughput.guideWeight * lightThroughput.guideWeight * cosines;
			mAssert(!isnan(newVal));
			write(m_targets[RenderTargets::LIGHTNESS], pixel, prev+ei::Vec4{newVal, 0.0f, 0.0f, 0.0f});
		}
		// Position, Normal and Albedo are handled by the random hit contribution.
	}

	// Handle contribution of random hit events
	__host__ __device__ void contribute(Pixel pixel,
										const Throughput & viewThroughput,
										const ei::Vec3& radiance,
										const ei::Vec3& position,
										const ei::Vec3& normal,
										const ei::Vec3& albedo
	) {
		using namespace scene::textures;
		if(is_valid(m_targets[RenderTargets::RADIANCE])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::RADIANCE], pixel);
			ei::Vec3 newVal = viewThroughput.weight * radiance;
			mAssert(!isnan(newVal.x) && !isnan(newVal.y) && !isnan(newVal.z));
			write(m_targets[RenderTargets::RADIANCE], pixel, prev+ei::Vec4{newVal, 0.0f});
		}
		if(is_valid(m_targets[RenderTargets::POSITION])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::POSITION], pixel);
			ei::Vec3 newVal = viewThroughput.guideWeight * position;
			mAssert(!isnan(newVal.x) && !isnan(newVal.y) && !isnan(newVal.z));
			write(m_targets[RenderTargets::POSITION], pixel, prev+ei::Vec4{newVal, 0.0f});
		}
		if(is_valid(m_targets[RenderTargets::NORMAL])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::NORMAL], pixel);
			ei::Vec3 newVal = viewThroughput.guideWeight * normal;
			mAssert(!isnan(newVal.x) && !isnan(newVal.y) && !isnan(newVal.z));
			write(m_targets[RenderTargets::NORMAL], pixel, prev+ei::Vec4{newVal, 0.0f});
		}
		if(is_valid(m_targets[RenderTargets::ALBEDO])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::ALBEDO], pixel);
			ei::Vec3 newVal = viewThroughput.guideWeight * albedo;
			mAssert(!isnan(newVal.x) && !isnan(newVal.y) && !isnan(newVal.z));
			write(m_targets[RenderTargets::ALBEDO], pixel, prev+ei::Vec4{newVal, 0.0f});
		}
		if(is_valid(m_targets[RenderTargets::LIGHTNESS])) {
			ei::Vec4 prev = read(m_targets[RenderTargets::LIGHTNESS], pixel);
			float newVal = viewThroughput.guideWeight * avg(radiance);
			mAssert(!isnan(newVal));
			write(m_targets[RenderTargets::LIGHTNESS], pixel, prev+ei::Vec4{newVal, 0.0f, 0.0f, 0.0f});
		}
	}

	__host__ __device__ int get_width() const { return m_resolution.x; }
	__host__ __device__ int get_height() const { return m_resolution.y; }
	__host__ __device__ ei::IVec2 get_resolution() const { return m_resolution; }
};

// Kind of a multiple-platform multiple-render-target.
class OutputHandler {
public:
	OutputHandler(u16 width, u16 height, OutputValue targets);

	// Allocate and clear the buffers. Which buffers are returned depents on the
	// 'targets' which where set in the constructor.
	template < Device dev >
	RenderBuffer<dev> begin_iteration(bool reset);

	// Do some finalization, like variance computations
	template < Device dev >
	void end_iteration();

	void set_targets(OutputValue targets);


	// Get the formated output of one quantity for the purpose of exporting screenshots.
	// which: The quantity to export. Causes an error if the quantity is not recorded.
	// exportFormat: The format of the pixels in the vector (includes elementary type and number of channels).
	// exportSRgb: Convert the values from linear to sRGB before packing the data into the exportFormat.
	scene::textures::CpuTexture get_data(OutputValue which, scene::textures::Format exportFormat, bool exportSRgb);

	// Get the actual texture data (read only) without converting to another format
	scene::textures::ConstTextureDevHandle_t<Device::CPU> get_data(OutputValue which);

	int get_current_iteration() const noexcept { return m_iteration; }

	int get_width() const { return m_width; }
	int get_height() const { return m_height; }
	int get_num_pixels() const { return m_width * m_height; }
	ei::IVec2 get_resolution() const { return {m_width, m_height}; }
	static scene::textures::Format get_target_format(OutputValue which);
private:
	// In each block either none, m_iter... only, or all three are defined.
	// If variances is required all three will be used and m_iter resets every iteration.
	// Otherwise m_iter contains the cumulative (non-normalized) radiance.
	scene::textures::Texture m_cumulativeTex[5];		// Accumulate the property (normalized by iteration count if variances are used)
	scene::textures::Texture m_iterationTex[5];			// Gets reset to 0 at the begin of each iteration (only required for variance)
	scene::textures::Texture m_cumulativeVarTex[5];		// Accumulate the variance
	OutputValue m_targets;
	int m_iteration;			// Number of completed iterations / index of current one
	int m_width;
	int m_height;


	void update_variance_cuda(scene::textures::TextureDevHandle_t<Device::CUDA> iterTex,
							  scene::textures::TextureDevHandle_t<Device::CUDA> cumTex,
							  scene::textures::TextureDevHandle_t<Device::CUDA> varTex);

	static __host__ __device__ void
	update_variance(scene::textures::TextureDevHandle_t<CURRENT_DEV> iterTex,
					scene::textures::TextureDevHandle_t<CURRENT_DEV> cumTex,
					scene::textures::TextureDevHandle_t<CURRENT_DEV> varTex,
					int x, int y, float iteration
	) {
		ei::Vec3 cum { read(cumTex, Pixel{x,y}) };
		ei::Vec3 iter { read(iterTex, Pixel{x,y}) };
		ei::Vec3 var { read(varTex, Pixel{x,y}) };
		// Use a stable addition scheme for the variance
		ei::Vec3 diff = iter - cum;
		cum += diff / ei::max(1.0f, iteration);
		var += diff * (iter - cum);
		write(cumTex, Pixel{x,y}, {cum, 0.0f});
		write(varTex, Pixel{x,y}, {var, 0.0f});
	}
	friend __global__ void update_variance_kernel(scene::textures::TextureDevHandle_t<Device::CUDA> iterTex,
								scene::textures::TextureDevHandle_t<Device::CUDA> cumTex,
								scene::textures::TextureDevHandle_t<Device::CUDA> varTex,
								float iteration);
};

}} // namespace mufflon::renderer
