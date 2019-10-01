#include "pt_params.hpp"
#include "pt_common.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_target.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__device__ int queueCounter;

__global__ static void sample_pt(PtTargets::template RenderBufferType<Device::CUDA> outputBuffer,
								 scene::SceneDescriptor<Device::CUDA>* scene,
								 math::Rng* rngs, PtParameters params) {
	const auto WIDTH = outputBuffer.get_width();
	const auto PIXELS = outputBuffer.get_num_pixels();
	int pixel = ::atomicAdd(&queueCounter, 1);
	while(pixel < PIXELS) {
		Pixel coord{ pixel % WIDTH, pixel / WIDTH };
#ifdef __CUDA_ARCH__
		pt_sample(outputBuffer, *scene, params, coord, rngs[pixel]);
#endif // __CUDA_ARCH__
		pixel = ::atomicAdd(&queueCounter, 1);
	}
}

namespace gpupt_detail {

cudaError_t call_kernel(PtTargets::template RenderBufferType<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const PtParameters& params) {
	int minGridSize;
	int blockSize;
	cuda::check_error(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sample_pt, 0));

	const dim3 blockDims{
		16u,
		static_cast<u32>(1 + (blockSize - 1) / 16),
		1u
	};
	// TODO: what are the optimal values here?
	const dim3 gridDims{
		1u + static_cast<u32>(std::max(1, outputBuffer.get_width() / 4) - 1) / blockDims.x,
		1u + static_cast<u32>(std::max(1, outputBuffer.get_height() / 4) - 1) / blockDims.y,
		1u
	};

	{
		void* ptr = nullptr;
		cuda::check_error(::cudaGetSymbolAddress(&ptr, queueCounter));
		cuda::check_error(::cudaMemset(ptr, 0, sizeof(int)));
	}

	sample_pt<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
									   rngs, params);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

__global__ static void init_rng(u32 num, int seed, math::Rng* rngs) {
	u32 idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < num) {
		rngs[idx] = math::Rng{ idx + seed };
	}
}
void init_rngs(u32 num, int seed, math::Rng* rngs) {
	dim3 blockDims { 256u, 1u, 1u };
	dim3 gridDims { (num + 255u) / 256u, 1u, 1u };
	init_rng<<<gridDims, blockDims>>>(num, seed, rngs);
	//cudaDeviceSynchronize();
	cuda::check_error(cudaGetLastError());
}

} // namespace gpupt_detail

}} // namespace mufflon::renderer
