#include "pt_params.hpp"
#include "pt_common.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/parameter.hpp"
#include "core/renderer/targets/render_target.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cooperative_groups.h>

using namespace mufflon::scene::lights;
using namespace cooperative_groups;

namespace mufflon { namespace renderer {

// Contains the current warp grid index for the next warp
__device__ int ptWarpGridIndex;

// These are alternative versions: one without any regeneration, and the other with "dumb" per-thread regeneration
#if 0
__global__ static void sample_pt(PtTargets::template RenderBufferType<Device::CUDA> outputBuffer,
								 scene::SceneDescriptor<Device::CUDA>* scene,
								 math::Rng* rngs, PtParameters params) {
	const Pixel coord{
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

__global__ static void sample_pt_regen(PtTargets::template RenderBufferType<Device::CUDA> outputBuffer,
									  scene::SceneDescriptor<Device::CUDA>* scene,
									  math::Rng* rngs, PtParameters params) {
	const auto WIDTH = outputBuffer.get_width();
	const auto PIXELS = outputBuffer.get_num_pixels();
	int pixel = ::atomicAdd(&ptWarpGridIndex, 1);
	while(pixel < PIXELS) {
		Pixel coord{ pixel % WIDTH, pixel / WIDTH };
#ifdef __CUDA_ARCH__
			pt_sample(outputBuffer, *scene, params, coord, rngs[pixel]);
#endif // __CUDA_ARCH__
		pixel = ::atomicAdd(&ptWarpGridIndex, 1);
	}
}
#endif // 0

__global__ static void sample_pt_warp(PtTargets::template RenderBufferType<Device::CUDA> outputBuffer,
								 scene::SceneDescriptor<Device::CUDA>* scene,
								 math::Rng* rngs, PtParameters params) {
	// To increase coherency, we group pixels into warp-sized blocks that are pulled from a queue
	// (the queue in our case is a simple atomic being incremented, and the blocks are numbered
	// row-major)
	auto warp = coalesced_threads();

	// The blocks are 8 pixels wide and as high as necessary (assuming warp-size % 8 == 0)
	static constexpr int WARP_X_RES = 8;
	const int WARP_Y_RES = warp.size() / WARP_X_RES;
	const auto WIDTH = outputBuffer.get_width();
	const auto HEIGHT = outputBuffer.get_height();
	const auto PIXELS = outputBuffer.get_num_pixels();
	const auto WARP_GRID_COUNT_X = 1 + (WIDTH - 1) / WARP_X_RES;
	const auto WARP_GRID_COUNT_Y = 1 + (HEIGHT - 1) / WARP_Y_RES;
	const auto WARP_GRID_COUNT = WARP_GRID_COUNT_X * WARP_GRID_COUNT_Y;

	// The warp leader fetches a new block index and broadcasts it to all others
	int gridIndex;
	if(warp.thread_rank() == 0)
		gridIndex = ::atomicAdd(&ptWarpGridIndex, 1);
	gridIndex = warp.shfl(gridIndex, 0);

	// As long as there are blocks not computed, we fetch new ones
	while(gridIndex < WARP_GRID_COUNT) {
		// Compute the pixel from the grid index and the local thread rank
		const auto gridXIndex = gridIndex % WARP_GRID_COUNT_X;
		const auto gridYIndex = gridIndex / WARP_GRID_COUNT_X;
		const Pixel warpFirstPixel{ gridXIndex * WARP_X_RES, gridYIndex * WARP_Y_RES };
		const Pixel innerWarpPixel{ warp.thread_rank() % WARP_X_RES, warp.thread_rank() / WARP_X_RES };
		const Pixel coord = warpFirstPixel + innerWarpPixel;
		const int pixel = coord.x + coord.y * WIDTH;

		// Normal PT - ignore pixels out-of-bounds
		if(coord.x < WIDTH && coord.y < HEIGHT) {
#ifdef __CUDA_ARCH__
			pt_sample(outputBuffer, *scene, params, coord, rngs[pixel]);
#endif // __CUDA_ARCH__
		}

		if(warp.thread_rank() == 0)
			gridIndex = ::atomicAdd(&ptWarpGridIndex, 1);
		gridIndex = warp.shfl(gridIndex, 0);
	}
}

namespace gpupt_detail {

cudaError_t call_kernel(PtTargets::template RenderBufferType<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						math::Rng* rngs, const PtParameters& params) {

	{
		// Reset the grid index
		void* ptr = nullptr;
		cuda::check_error(::cudaGetSymbolAddress(&ptr, ptWarpGridIndex));
		cuda::check_error(::cudaMemset(ptr, 0, sizeof(int)));
	}

	int minGridSize;
	int blockSize;
	cuda::check_error(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sample_pt_warp, 0));

	const dim3 blockDims{
		16u,
		static_cast<u32>(1 + (blockSize - 1) / 16),
		1u
	};

	// TODO: what are the optimal values here? 16x less threads for regeneration seemed
	// to do well on a GTX 980Ti
	const int GRID_X_FACTOR = 4;
	const int GRID_Y_FACTOR = 4;
	const dim3 gridDims{
		1u + static_cast<u32>(std::max(1, outputBuffer.get_width() / GRID_X_FACTOR) - 1) / blockDims.x,
		1u + static_cast<u32>(std::max(1, outputBuffer.get_height() / GRID_Y_FACTOR) - 1) / blockDims.y,
		1u
	};
	sample_pt_warp<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
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
