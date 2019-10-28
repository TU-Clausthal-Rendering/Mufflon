#include "ss_pt_common.hpp"
#include "ss_pt_params.hpp"
#include "ss_importance_gathering_pt.hpp"
#include "core/cuda/error.hpp"
#include <device_launch_parameters.h>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette {

using namespace ss;

__global__ static void init_rng(u32 num, int seed, math::Rng* rngs) {
	u32 idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < num) {
		rngs[idx] = math::Rng{ idx + seed };
	}
}

namespace gpusssil_detail {

cudaError_t call_kernel_sample(const SilhouetteTargets::template RenderBufferType<Device::CUDA>& outputBuffer,
							   scene::SceneDescriptor<Device::CUDA>* scene,
							   math::Rng* rngs, const SilhouetteParameters& params) {
	// TODO!

	/*int minGridSize;
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
											rngs, params);*/

	::cudaDeviceSynchronize();
	return ::cudaGetLastError();
}

cudaError_t call_kernel_postprocess(const SilhouetteTargets::template RenderBufferType<Device::CUDA>& outputBuffer,
									scene::SceneDescriptor<Device::CUDA>* scene,
									math::Rng* rngs, const SilhouetteParameters& params) {
	// TODO!
	::cudaDeviceSynchronize();
	return ::cudaGetLastError();
}

void init_rngs(u32 num, int seed, math::Rng* rngs) {
	dim3 blockDims{ 256u, 1u, 1u };
	dim3 gridDims{ (num + 255u) / 256u, 1u, 1u };
	init_rng<<<gridDims, blockDims>>>(num, seed, rngs);
	//cudaDeviceSynchronize();
	cuda::check_error(cudaGetLastError());
}

} // namespace gpusssil_detail

}}}} // namespace mufflon::renderer::decimaters::silhouette