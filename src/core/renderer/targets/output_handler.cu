#include "output_handler_variance.hpp"
#include "core/cuda/error.hpp"
#include <device_launch_parameters.h>

namespace mufflon { namespace renderer {

template < class PixelType, bool ReduceMoments >
__global__ void update_iter_kernel(ConstRenderTargetBuffer<Device::CUDA, PixelType> iterTarget,
								   RenderTargetBuffer<Device::CUDA, float> cumTarget,
								   RenderTargetBuffer<Device::CUDA, float> varTarget,
								   int numChannels,
								   int width, int height,
								   float iteration) {
#ifdef __CUDA_ARCH__
	int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	if(x < width && y < height)
		output_handler_details::UpdateIter<PixelType, ReduceMoments>::f(iterTarget,
				cumTarget, varTarget, x, y, numChannels, width, iteration);
#endif
}

namespace output_handler_details {

template < class PixelType, bool ReduceMoments >
void update_iter_cuda(ConstRenderTargetBuffer<Device::CUDA, PixelType> iterTarget,
					  RenderTargetBuffer<Device::CUDA, float> cumTarget,
					  RenderTargetBuffer<Device::CUDA, float> varTarget,
					  int numChannels, int width, int height, int iteration) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
		(height + dimBlock.y - 1) / dimBlock.y);
	update_iter_kernel<PixelType, ReduceMoments><<<dimGrid, dimBlock>>>(
		iterTarget, cumTarget, varTarget, numChannels, width, height, float(iteration));
}

template void update_iter_cuda<float, true>(ConstRenderTargetBuffer<Device::CUDA, float>,
											RenderTargetBuffer<Device::CUDA, float>,
											RenderTargetBuffer<Device::CUDA, float>,
											int, int, int, int);

template void update_iter_cuda<float, false>(ConstRenderTargetBuffer<Device::CUDA, float>,
											 RenderTargetBuffer<Device::CUDA, float>,
											 RenderTargetBuffer<Device::CUDA, float>,
											 int, int, int, int);

template void update_iter_cuda<i32, true>(ConstRenderTargetBuffer<Device::CUDA, i32>,
										  RenderTargetBuffer<Device::CUDA, float>,
										  RenderTargetBuffer<Device::CUDA, float>,
										  int, int, int, int);

template void update_iter_cuda<i32, false>(ConstRenderTargetBuffer<Device::CUDA, i32>,
										   RenderTargetBuffer<Device::CUDA, float>,
										   RenderTargetBuffer<Device::CUDA, float>,
										   int, int, int, int);

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

} // namespace output_handler_details

}} // namespace mufflon::renderer