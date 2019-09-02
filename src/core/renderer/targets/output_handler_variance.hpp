#pragma once

#include "render_target.hpp"

namespace mufflon { namespace renderer { namespace output_handler_details {

template < class PixelType, bool ReduceMoments >
struct UpdateIter {
	static CUDA_FUNCTION void
	f(ConstRenderTargetBuffer<CURRENT_DEV, PixelType> iterTarget,
	  RenderTargetBuffer<CURRENT_DEV, float> cumTarget,
	  RenderTargetBuffer<CURRENT_DEV, float> varTarget,
	  int x, int y, int num_channels, int width, float iteration) {
		for(int c = 0; c < num_channels; ++c) {
			int idx = c + (x + y * width) * num_channels;
			auto iter = static_cast<float>(cuda::atomic_load<CURRENT_DEV, PixelType>(iterTarget[idx]));
			auto cum = cuda::atomic_load<CURRENT_DEV, float>(cumTarget[idx]);
			// Use a stable addition scheme for the mean and variance
			auto diff = iter - cum;
			cum += diff / ei::max(1.0f, iteration+1);
			cuda::atomic_exchange<CURRENT_DEV>(cumTarget[idx], cum);
			if(varTarget) {
				float var = cuda::atomic_load<CURRENT_DEV, float>(varTarget[idx]);
				var += diff * (iter - cum);
				cuda::atomic_exchange<CURRENT_DEV>(varTarget[idx], var);
			}
		}
	}
};

template < class PixelType >
struct UpdateIter<PixelType, true> {
	static CUDA_FUNCTION void
	f(ConstRenderTargetBuffer<CURRENT_DEV, PixelType> iterTarget,
	  RenderTargetBuffer<CURRENT_DEV, float> cumTarget,
	  RenderTargetBuffer<CURRENT_DEV, float> varTarget,
	  int x, int y, int num_channels, int width, float iteration) {
		for(int c = 0; c < num_channels; ++c) {
			int idx = c + (x + y * width) * num_channels;
			auto iter = static_cast<float>(cuda::atomic_load<CURRENT_DEV, PixelType>(iterTarget[idx]));
			cuda::atomic_exchange<CURRENT_DEV>(cumTarget[idx], iter);
			if(varTarget) {
				auto cum = cuda::atomic_load<CURRENT_DEV, float>(varTarget[idx]);
				// Use a stable addition scheme for the mean
				auto diff = iter - cum;
				cum += diff / ei::max(1.0f, iteration+1);
				cuda::atomic_exchange<CURRENT_DEV>(varTarget[idx], cum);
			}
		}
	}
};

}}} // namespace mufflon::renderer::output_handler_details