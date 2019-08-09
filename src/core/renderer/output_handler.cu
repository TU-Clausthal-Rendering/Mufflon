#include "output_handler.hpp"
#include <device_launch_parameters.h>

namespace mufflon { namespace renderer {

__global__ void update_variance_kernel(ConstRenderTarget<Device::CUDA> iterTarget,
								RenderTarget<Device::CUDA> cumTarget,
								RenderTarget<Device::CUDA> varTarget,
								int numChannels,
								int width, int height,
								float iteration
) {
#ifdef __CUDA_ARCH__
	int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	if(x < width && y < height)
		OutputHandler::update_variance(iterTarget, cumTarget, varTarget, x, y, 
			numChannels, width, iteration);
#endif
}

void OutputHandler::update_variance_cuda(ConstRenderTarget<Device::CUDA> iterTarget,
							RenderTarget<Device::CUDA> cumTarget,
							RenderTarget<Device::CUDA> varTarget,
							int numChannels
) {
	dim3 dimBlock(16,16);
	dim3 dimGrid((m_width + dimBlock.x-1) / dimBlock.x,
					(m_height + dimBlock.y-1) / dimBlock.y);
	update_variance_kernel<<<dimGrid,dimBlock>>>(
		iterTarget, cumTarget, varTarget, numChannels, m_width, m_height, float(m_iteration));
}

__device__ u32 *s_nan_counter;

u32* get_cuda_nan_counter_ptr_and_set_zero() {
	constexpr u32 zero = 0;
	void* ptr = nullptr;
	cuda::check_error(::cudaGetSymbolAddress(&ptr, s_nan_counter));
	cuda::check_error(::cudaMemcpyToSymbolAsync(s_nan_counter, &zero, sizeof(zero),
												0u, ::cudaMemcpyHostToDevice));
	return reinterpret_cast<u32*>(ptr);
}

u32 get_cuda_nan_counter_value() {
	u32 counter = 0;
	cuda::check_error(::cudaMemcpyFromSymbolAsync(&counter, s_nan_counter, sizeof(counter),
												  0u, ::cudaMemcpyDeviceToHost));
	return counter;
}

}} // namespace mufflon::renderer