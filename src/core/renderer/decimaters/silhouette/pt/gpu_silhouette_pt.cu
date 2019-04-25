#include "silhouette_importance_gathering_pt.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mufflon { namespace renderer { namespace decimaters {

using namespace silhouette;

__global__ static void silhouette_kernel_pt(RenderBuffer<Device::CUDA> outputBuffer,
											scene::SceneDescriptor<Device::CUDA>* scene,
											const u32* seeds, SilhouetteParameters params,
											Importances<Device::CUDA>** importances,
											DeviceImportanceSums<Device::CUDA>* sums) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();

	math::Rng rng(seeds[pixel]);
#ifdef __CUDA_ARCH__
	sample_importance(outputBuffer, *scene, params, coord, rng, importances, sums);
#endif // __CUDA_ARCH__
}

namespace gpusil_details {


cudaError_t call_importance_kernel(const dim3& gridDims, const dim3& blockDims,
								   renderer::RenderBuffer<Device::CUDA>&& outputBuffer,
								   scene::SceneDescriptor<Device::CUDA>* scene,
								   const u32* seeds, const SilhouetteParameters& params,
								   Importances<Device::CUDA>** importances,
								   DeviceImportanceSums<Device::CUDA>* sums) {
	silhouette_kernel_pt<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
												  seeds, params, importances, sums);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

} // namespace gpusil_details

}}} // namespace mufflon::renderer::decimaters