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
								 const u32* seeds, PtParameters params) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();

	math::Rng rng(seeds[pixel]);
#ifdef __CUDA_ARCH__
	pt_sample(outputBuffer, *scene, params, coord, rng);
#endif // __CUDA_ARCH__
}

namespace gpupt_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						const u32* seeds, const PtParameters& params) {
	sample_pt<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
									   seeds, params);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

} // namespace gpupt_detail

}} // namespace mufflon::renderer
