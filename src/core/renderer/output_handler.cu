#include "output_handler.hpp"
#include <device_launch_parameters.h>

using namespace mufflon::scene::textures;

namespace mufflon { namespace renderer {

__global__ void update_variance_kernel(TextureDevHandle_t<Device::CUDA> iterTex,
								TextureDevHandle_t<Device::CUDA> cumTex,
								TextureDevHandle_t<Device::CUDA> varTex,
								float iteration
) {
#ifdef __CUDA_ARCH__
	int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	if(x < iterTex.width && y < iterTex.height)
		OutputHandler::update_variance(iterTex, cumTex, varTex, x, y, iteration);
#endif
}

void OutputHandler::update_variance_cuda(TextureDevHandle_t<Device::CUDA> iterTex,
							TextureDevHandle_t<Device::CUDA> cumTex,
							TextureDevHandle_t<Device::CUDA> varTex
) {
	dim3 dimBlock(16,16);
	dim3 dimGrid((m_width + dimBlock.x-1) / dimBlock.x,
					(m_height + dimBlock.y-1) / dimBlock.y);
	update_variance_kernel<<<dimGrid,dimBlock>>>(
		iterTex, cumTex, varTex, float(m_iteration));
}

}} // namespace mufflon::renderer