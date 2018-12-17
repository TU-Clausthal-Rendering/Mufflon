#include "light_tree.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/light_medium.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mufflon { namespace scene { namespace lights {

__global__ void cuda_set_medium(SceneDescriptor<Device::CUDA> scene, LightSubTree posLights,
								const u32 NODE_COUNT) {
	unsigned long long nodeIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(nodeIdx < NODE_COUNT) {
		const LightSubTree::Node& node = *posLights.get_node((NODE_COUNT - nodeIdx) * static_cast<u32>(sizeof(LightSubTree::Node)));
		if(nodeIdx * 2 > posLights.lightCount) {
			// Single light node;
			mAssert(node.right.type < static_cast<u16>(LightType::NUM_LIGHTS));
#ifdef __CUDA_ARCH__
			set_light_medium(&posLights.memory[node.right.offset], static_cast<LightType>(node.right.type), scene);
#endif // __CUDA_ARCH__
		} else {
			// Dual light node
			mAssert(node.left.type < static_cast<u16>(LightType::NUM_LIGHTS));
			mAssert(node.right.type < static_cast<u16>(LightType::NUM_LIGHTS));
#ifdef __CUDA_ARCH__
			set_light_medium(&posLights.memory[node.left.offset], static_cast<LightType>(node.left.type), scene);
			set_light_medium(&posLights.memory[node.right.offset], static_cast<LightType>(node.right.type), scene);
#endif // __CUDA_ARCH__
		}
	}
}

namespace lighttree_detail {

void update_media_cuda(const SceneDescriptor<Device::CUDA>& scene, const LightSubTree& posLights) {
	const u32 NODE_COUNT = static_cast<u32>(get_num_internal_nodes(posLights.lightCount));

	int blockSize = 256;
	int gridSize;
	if(NODE_COUNT == 0)
		gridSize = 1;
	else
		gridSize = (NODE_COUNT - 1) / blockSize + 1;

	cuda_set_medium<<<gridSize, blockSize>>>(scene, posLights, NODE_COUNT);
	cuda::check_error(cudaGetLastError());
}

} // namespace lighttree_detail

}}} // namespace mufflon::scene::lights