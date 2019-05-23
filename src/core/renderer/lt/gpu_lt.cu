#include "lt_params.hpp"
#include "lt_common.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/parameter.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void sample_lt(RenderBuffer<Device::CUDA> outputBuffer,
								 scene::SceneDescriptor<Device::CUDA>* scene,
								 math::Rng* rngs, LtParameters params) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int idx = coord.x + coord.y * outputBuffer.get_width();

#ifdef __CUDA_ARCH__
	lt_sample(outputBuffer, *scene, params, idx, rngs[idx]);
#endif // __CUDA_ARCH__
}

namespace gpult_detail {

cudaError_t call_kernel(RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const LtParameters& params) {
	int minGridSize;
	int blockSize;
	cuda::check_error(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sample_lt, 0));

	const dim3 blockDims{
		16u,
		static_cast<u32>(1 + (blockSize - 1) / 16),
		1u
	};
	const dim3 gridDims{
		1u + static_cast<u32>(outputBuffer.get_width() - 1) / blockDims.x,
		1u + static_cast<u32>(outputBuffer.get_height() - 1) / blockDims.y,
		1u
	};

	sample_lt<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
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
	//cudaDeviceSynchronize();
	cuda::check_error(cudaGetLastError());
}

} // namespace gpult_detail

}} // namespace mufflon::renderer
