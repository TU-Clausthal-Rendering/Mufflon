#pragma once

#include "path_util.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/cputexture.hpp"
#include "core/scene/textures/interface.hpp"
#include "core/cuda/cuda_utils.hpp"
#include "util/flag.hpp"
#include "util/string_view.hpp"
#include "core/math/sample_types.hpp"

namespace mufflon { namespace renderer {

struct RenderTargets {
	static constexpr u32 RADIANCE = 0u;
	static constexpr u32 POSITION = 1u;
	static constexpr u32 ALBEDO = 2u;
	static constexpr u32 NORMAL = 3u;
	static constexpr u32 LIGHTNESS = 4u;

	static constexpr int NUM_CHANNELS[] = { 3, 3, 3, 3, 1 };
};

struct OutputValue : util::Flags<u32> {
	static constexpr u32 VARIANCE_OFFSET = 8u;
	static constexpr u32 VARIANCE_MASK = 0xFFFFFFFF << VARIANCE_OFFSET;
	static constexpr u32 make_variance(u32 target) noexcept { return target << VARIANCE_OFFSET; }
	static constexpr u32 remove_variance(u32 varianceTarget) noexcept { return varianceTarget >> VARIANCE_OFFSET; }

	static constexpr u32 RADIANCE = 0x0001;			// Output of radiance (standard output of a renderer)
	static constexpr u32 POSITION = 0x0002;			// Output of positions (customly integrated over paths)
	static constexpr u32 ALBEDO = 0x0004;			// Output of albedo (customly integrated over paths)
	static constexpr u32 NORMAL = 0x0008;			// Output of signed normals (customly integrated over paths)
	static constexpr u32 LIGHTNESS = 0x0010;		// Output of multiplied paths weights (from the custom integration)

	static constexpr u32 RADIANCE_VAR = RADIANCE << VARIANCE_OFFSET;		// Variance of radiance
	static constexpr u32 POSITION_VAR = POSITION << VARIANCE_OFFSET;
	static constexpr u32 ALBEDO_VAR = ALBEDO << VARIANCE_OFFSET;
	static constexpr u32 NORMAL_VAR = NORMAL << VARIANCE_OFFSET;
	static constexpr u32 LIGHTNESS_VAR = LIGHTNESS << VARIANCE_OFFSET;

	static constexpr u32 iterator[] = { RADIANCE, POSITION, ALBEDO, NORMAL, LIGHTNESS };
	static constexpr std::size_t TARGET_COUNT = sizeof(iterator) / sizeof(*iterator);

	bool is_variance() const { return (mask & VARIANCE_MASK) != 0; }
	int get_quantity() const { return ei::ilog2(is_variance() ? int(mask >> 8) : int(mask)); }
};

inline StringView get_render_target_name(u32 target) {
	static StringView TARGET_NAMES[] = {
		"Radiance", "Position", "Albedo", "Normal", "Lightness"
	};
	static_assert(sizeof(TARGET_NAMES) / sizeof(*TARGET_NAMES) == OutputValue::TARGET_COUNT,
				  "Inequal number of targets and target names");
	mAssert(target < OutputValue::TARGET_COUNT);
	return TARGET_NAMES[target];
}

template < Device dev >
using RenderTarget = ArrayDevHandle_t<dev, cuda::Atomic<dev, float>>;
template < Device dev >
using ConstRenderTarget = ConstArrayDevHandle_t<dev, cuda::Atomic<dev, float>>;

template < Device dev >
struct RenderBuffer {
	// The following texture handles may contain iteration only or all iterations summed
	// information. The meaning of the variables is only known to the OutputHandler.
	// The renderbuffer only needs to add values to all defined handles.
	RenderTarget<dev> m_targets[OutputValue::TARGET_COUNT] = {};
	ei::IVec2 m_resolution;

	__host__ __device__ float check_nan(float x) {
		if(isnan(x)) {
#ifndef  __CUDA_ARCH__
			logWarning("[RenderBuffer] Detected NaN on output. Returning 0 instead.");
#endif // ! __CUDA_ARCH__
			return 0.0f;
		}
		return x;
	}
	__host__ __device__ ei::Vec3 check_nan(const ei::Vec3& x) {
		if(isnan(x.x) || isnan(x.y) || isnan(x.z)) {
#ifndef  __CUDA_ARCH__
			logWarning("[RenderBuffer] Detected NaN on output. Returning 0 instead.");
#endif // ! __CUDA_ARCH__
			return ei::Vec3{ 0.0f };
		}
		return x;
	}

	bool is_target_enabled(u32 target) const noexcept {
		return (target < OutputValue::TARGET_COUNT) &&  scene::textures::is_valid(m_targets[target]);
	}

	/*
	 * Handle contribution of connection and merge events
	 * value: The radiance estimate from the event. This can be the BxDF (merge) or
	 *		BxDF * BxDF / distSq for connections.
	 */
	__host__ __device__ void contribute(Pixel pixel,
										const math::Throughput& viewThroughput,
										const math::Throughput& lightThroughput,
										float cosines, const ei::Vec3& value
	) {
		if(pixel.x < 0 || pixel.x >= m_resolution.x || pixel.y < 0 || pixel.y >= m_resolution.y)
			return;
		if(m_targets[RenderTargets::RADIANCE]) {
			ei::Vec3 newVal = viewThroughput.weight * lightThroughput.weight * value * cosines;
			newVal = check_nan(newVal);
			size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx  ], newVal.x);
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx+1], newVal.y);
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx+2], newVal.z);
		}
		if(m_targets[RenderTargets::LIGHTNESS]) {
			float newVal = viewThroughput.guideWeight * lightThroughput.guideWeight * cosines;
			newVal = check_nan(newVal);
			size_t idx = pixel.x + pixel.y * m_resolution.x;
			cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx], newVal);
		}
		// Position, Normal and Albedo are handled by the random hit contribution.
	}

	// Handle contribution of random hit events
	__host__ __device__ void contribute(Pixel pixel,
										const math::Throughput & viewThroughput,
										const ei::Vec3& radiance,
										const ei::Vec3& position,
										const ei::Vec3& normal,
										const ei::Vec3& albedo
	) {
		if(pixel.x < 0 || pixel.x >= m_resolution.x || pixel.y < 0 || pixel.y >= m_resolution.y)
			return;
		if(m_targets[RenderTargets::RADIANCE]) {
			ei::Vec3 newVal = viewThroughput.weight * radiance;
			newVal = check_nan(newVal);
			size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx  ], newVal.x);
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx+1], newVal.y);
			cuda::atomic_add<dev>(m_targets[RenderTargets::RADIANCE][idx+2], newVal.z);
		}
		if(m_targets[RenderTargets::POSITION]) {
			ei::Vec3 newVal = viewThroughput.guideWeight * position;
			newVal = check_nan(newVal);
			size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
			cuda::atomic_add<dev>(m_targets[RenderTargets::POSITION][idx  ], newVal.x);
			cuda::atomic_add<dev>(m_targets[RenderTargets::POSITION][idx+1], newVal.y);
			cuda::atomic_add<dev>(m_targets[RenderTargets::POSITION][idx+2], newVal.z);
		}
		if(m_targets[RenderTargets::NORMAL]) {
			ei::Vec3 newVal = viewThroughput.guideWeight * normal;
			newVal = check_nan(newVal);
			size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
			cuda::atomic_add<dev>(m_targets[RenderTargets::NORMAL][idx  ], newVal.x);
			cuda::atomic_add<dev>(m_targets[RenderTargets::NORMAL][idx+1], newVal.y);
			cuda::atomic_add<dev>(m_targets[RenderTargets::NORMAL][idx+2], newVal.z);
		}
		if(m_targets[RenderTargets::ALBEDO]) {
			ei::Vec3 newVal = viewThroughput.guideWeight * albedo;
			newVal = check_nan(newVal);
			size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
			cuda::atomic_add<dev>(m_targets[RenderTargets::ALBEDO][idx  ], newVal.x);
			cuda::atomic_add<dev>(m_targets[RenderTargets::ALBEDO][idx+1], newVal.y);
			cuda::atomic_add<dev>(m_targets[RenderTargets::ALBEDO][idx+2], newVal.z);
		}
		if(m_targets[RenderTargets::LIGHTNESS]) {
			float newVal = viewThroughput.guideWeight * avg(radiance);
			newVal = check_nan(newVal);
			size_t idx = pixel.x + pixel.y * m_resolution.x;
			cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx], newVal);
		}
	}

	__host__ __device__ void contribute(Pixel pixel, u32 target, const ei::Vec3& value) {
		if(pixel.x < 0 || pixel.x >= m_resolution.x || pixel.y < 0 || pixel.y >= m_resolution.y)
			return;
		if(m_targets[target]) {
			ei::Vec3 newVal = check_nan(value);
			if(target == RenderTargets::LIGHTNESS) {
				size_t idx = pixel.x + pixel.y * m_resolution.x;
				cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx], newVal.x);
			} else {
				size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
				cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx  ], newVal.x);
				cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx+1], newVal.y);
				cuda::atomic_add<dev>(m_targets[RenderTargets::LIGHTNESS][idx+2], newVal.z);
			}
		}
	}

	__host__ __device__ void set(Pixel pixel, u32 target, const ei::Vec3& value) {
		if(pixel.x < 0 || pixel.x >= m_resolution.x || pixel.y < 0 || pixel.y >= m_resolution.y)
			return;
		if(m_targets[target]) {
			ei::Vec3 newVal = check_nan(value);
			if(target == RenderTargets::LIGHTNESS) {
				size_t idx = pixel.x + pixel.y * m_resolution.x;
				cuda::atomic_exchange<dev>(m_targets[RenderTargets::LIGHTNESS][idx], newVal.x);
			} else {
				size_t idx = (pixel.x + pixel.y * m_resolution.x) * 3;
				cuda::atomic_exchange<dev>(m_targets[target][idx  ], newVal.x);
				cuda::atomic_exchange<dev>(m_targets[target][idx+1], newVal.y);
				cuda::atomic_exchange<dev>(m_targets[target][idx+2], newVal.z);
			}
		}
	}

	__host__ __device__ int get_width() const { return m_resolution.x; }
	__host__ __device__ int get_height() const { return m_resolution.y; }
	__host__ __device__ ei::IVec2 get_resolution() const { return m_resolution; }
	__host__ __device__ int get_num_pixels() const { return m_resolution.x * m_resolution.y; }
};

// Kind of a multiple-platform multiple-render-target.
class OutputHandler {
public:
	OutputHandler(u16 width, u16 height, OutputValue targets);

	// Allocate and clear the buffers. Which buffers are returned depents on the
	// 'targets' which where set in the constructor.
	template < Device dev >
	RenderBuffer<dev> begin_iteration(bool reset);
	template < Device dev1, Device dev2 >
	std::tuple<RenderBuffer<dev1>, RenderBuffer<dev2>> begin_iteration_hybrid(bool reset);

	// Do some finalization, like variance computations
	template < Device dev >
	void end_iteration();

	// For multi-device renderers: synchronizes the result split along the x-axis
	template < Device from, Device to >
	void sync_back(int ySplit);

	void set_targets(OutputValue targets);
	OutputValue get_target() const noexcept { return m_targets; }


	// Get the formated output of one quantity for the purpose of exporting screenshots.
	// which: The quantity to export. Causes an error if the quantity is not recorded.
	// The returned buffer is either Vec3 or float, depending on the number of channels in
	// the queried quantity.
	std::unique_ptr<float[]> get_data(OutputValue which);

	// Returns the value of a pixel as a Vec4, regardless of the underlying format
	ei::Vec4 get_pixel_value(OutputValue which, Pixel pixel);

	int get_current_iteration() const noexcept { return m_iteration; }

	int get_width() const { return m_width; }
	int get_height() const { return m_height; }
	int get_num_pixels() const { return m_width * m_height; }
	ei::IVec2 get_resolution() const { return {m_width, m_height}; }
private:
	// In each block either none, m_iter... only, or all three are defined.
	// If variances is required all three will be used and m_iter resets every iteration.
	// Otherwise m_iter contains the cumulative (non-normalized) radiance.
	GenericResource m_cumulativeTarget[OutputValue::TARGET_COUNT];			// Accumulate the property (normalized by iteration count if variances are used)
	GenericResource m_iterationTarget[OutputValue::TARGET_COUNT];			// Gets reset to 0 at the begin of each iteration (only required for variance)
	GenericResource m_cumulativeVarTarget[OutputValue::TARGET_COUNT];		// Accumulate the variance
	OutputValue m_targets;
	int m_iteration;			// Number of completed iterations / index of current one
	int m_width;
	int m_height;


	void update_variance_cuda(ConstRenderTarget<Device::CUDA> iterTarget,
							  RenderTarget<Device::CUDA> cumTarget,
							  RenderTarget<Device::CUDA> varTarget,
							  int numChannels);

	static __host__ __device__ void
	update_variance(ConstRenderTarget<CURRENT_DEV> iterTarget,
					RenderTarget<CURRENT_DEV> cumTarget,
					RenderTarget<CURRENT_DEV> varTarget,
					int x, int y, int numChannels, int width, float iteration
	) {
		for(int c = 0; c < numChannels; ++c) {
			int idx = c + (x + y * width) * numChannels;
			float iter = cuda::atomic_load<CURRENT_DEV, float>(iterTarget[idx]);
			float cum = cuda::atomic_load<CURRENT_DEV, float>(cumTarget[idx]);
			float var = cuda::atomic_load<CURRENT_DEV, float>(varTarget[idx]);
			// Use a stable addition scheme for the variance
			float diff = iter - cum;
			cum += diff / ei::max(1.0f, iteration);
			var += diff * (iter - cum);
			cuda::atomic_exchange<CURRENT_DEV>(cumTarget[idx], cum);
			cuda::atomic_exchange<CURRENT_DEV>(varTarget[idx], var);
		}
	}
	friend __global__ void update_variance_kernel(ConstRenderTarget<Device::CUDA> iterTex,
								RenderTarget<Device::CUDA> cumTex,
								RenderTarget<Device::CUDA> varTex,
								int numChannels,
								int width, int height,
								float iteration);
};

}} // namespace mufflon::renderer
