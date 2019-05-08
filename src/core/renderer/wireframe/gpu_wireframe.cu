#include "wireframe_params.hpp"
#include "wireframe_common.hpp"
#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ei/vector.hpp>
#include <random>

using namespace mufflon::scene::lights;

namespace mufflon {
namespace renderer {

__global__ static void wireframe_kernel(RenderBuffer<Device::CUDA> outputBuffer,
										scene::SceneDescriptor<Device::CUDA>* scene,
										const u32* seeds, WireframeParameters params) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();
	math::Rng rng(seeds[pixel]);

#ifdef __CUDA_ARCH__
	sample_wireframe(outputBuffer, *scene, params, rng, coord);
#endif // __CUDA_ARCH__
}

namespace gpuwireframe_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						const u32* seeds, const WireframeParameters& params) {
	wireframe_kernel<<<gridDims, blockDims>>>(std::move(outputBuffer), scene, seeds, params);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

} // namespace gpuwireframe_detail

}
} // namespace mufflon::renderer
