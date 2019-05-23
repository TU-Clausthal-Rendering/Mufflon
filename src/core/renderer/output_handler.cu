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

}} // namespace mufflon::renderer