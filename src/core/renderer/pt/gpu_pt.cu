#include "pt_params.hpp"
#include "pt_common.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/parameter.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void sample_pt(RenderBuffer<Device::CUDA> outputBuffer,
								 scene::SceneDescriptor<Device::CUDA>* scene,
								 math::Rng* rngs, PtParameters params) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();

#ifdef __CUDA_ARCH__
	pt_sample(outputBuffer, *scene, params, coord, rngs[pixel]);
#endif // __CUDA_ARCH__
}

namespace gpupt_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const PtParameters& params) {
	sample_pt<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
									   rngs, params);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

__global__ static void init_rng(u32 num, math::Rng* rngs) {
	u32 idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < num) {
		rngs[idx] = math::Rng{ idx };
	}
}
void init_rngs(u32 num, math::Rng* rngs) {
	dim3 blockDims { 256u, 1u, 1u };
	dim3 gridDims { (num + 255u) / 256u, 1u, 1u };
	init_rng<<<gridDims, blockDims>>>(num, rngs);
	cudaDeviceSynchronize();
	cuda::check_error(cudaGetLastError());
}

} // namespace gpupt_detail

}} // namespace mufflon::renderer
