#if 0
#include "silhouette_importance_gathering_pt.hpp"
#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

__global__ static void silhouette_kernel(pt::SilhouetteTargets::RenderBufferType<Device::CUDA> outputBuffer,
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
	//sample_importance(outputBuffer, *scene, params, coord, rng, importances, sums, impGrid);
#endif // __CUDA_ARCH__
}

__global__ static void impvis_kernel(pt::SilhouetteTargets::RenderBufferType<Device::CUDA> outputBuffer,
									 scene::SceneDescriptor<Device::CUDA>* scene,
									 const u32* seeds, Importances<Device::CUDA>** importances,
									 DeviceImportanceSums<Device::CUDA>* sums,
									 const float maxImportance) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();

	math::Rng rng(seeds[pixel]);
#ifdef __CUDA_ARCH__
	//sample_vis_importance(outputBuffer, *scene, coord, rng, importances, sums, maxImportance, impGrid);
#endif // __CUDA_ARCH__
}

namespace gpusil_details {


cudaError_t call_importance_kernel(const dim3& gridDims, const dim3& blockDims,
								   pt::SilhouetteTargets::RenderBufferType<Device::CUDA>&& outputBuffer,
								   scene::SceneDescriptor<Device::CUDA>* scene,
								   const u32* seeds, const SilhouetteParameters& params,
								   Importances<Device::CUDA>** importances,
								   DeviceImportanceSums<Device::CUDA>* sums) {
	silhouette_kernel<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
												  seeds, params, importances, sums);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t call_impvis_kernel(const dim3& gridDims, const dim3& blockDims,
							   pt::SilhouetteTargets::RenderBufferType<Device::CUDA>&& outputBuffer,
							   scene::SceneDescriptor<Device::CUDA>* scene,
							   const u32* seeds, Importances<Device::CUDA>** importances,
							   DeviceImportanceSums<Device::CUDA>* sums,
							   const float maxImportance) {
	impvis_kernel<<<gridDims, blockDims>>>(std::move(outputBuffer), scene,
										   seeds, importances, sums, maxImportance);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

} // namespace gpusil_details

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt
#endif // 0