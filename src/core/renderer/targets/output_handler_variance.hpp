#pragma once

#include "render_target.hpp"

namespace mufflon { namespace renderer { namespace output_handler_details {

template < class T >
CUDA_FUNCTION void
update_variance(ConstRenderTargetBuffer<CURRENT_DEV, T> iterTarget,
				RenderTargetBuffer<CURRENT_DEV, T> cumTarget,
				RenderTargetBuffer<CURRENT_DEV, T> varTarget,
				int x, int y, int num_channels, int width, float iteration) {
	for(int c = 0; c < num_channels; ++c) {
		int idx = c + (x + y * width) * num_channels;
		auto iter = cuda::atomic_load<CURRENT_DEV, T>(iterTarget[idx]);
		auto cum = cuda::atomic_load<CURRENT_DEV, T>(cumTarget[idx]);
		auto var = cuda::atomic_load<CURRENT_DEV, T>(varTarget[idx]);
		// Use a stable addition scheme for the variance
		auto diff = iter - cum;
		cum += diff / ei::max(1.0f, iteration);
		var += diff * (iter - cum);
		cuda::atomic_exchange<CURRENT_DEV>(cumTarget[idx], cum);
		cuda::atomic_exchange<CURRENT_DEV>(varTarget[idx], var);
	}
}

}}} // namespace mufflon::renderer::output_handler_details