#include "light_tree.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/light_medium.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>

namespace mufflon { namespace scene { namespace lights {

__global__ void cuda_set_medium(SceneDescriptor<Device::CUDA> scene, LightSubTree posLights,
								const u32 INTERNAL_NODE_COUNT, const u32 TREE_HEIGHT) {
	const u32 lightIdx = static_cast<u32>(threadIdx.x + blockDim.x * blockIdx.x);

#ifdef __CUDA_ARCH__
	if(lightIdx < posLights.lightCount) {
		// Special case: only a single light
		if(posLights.lightCount == 1u) {
			// Extract information from the root
			mAssert(posLights.root.type < static_cast<u16>(LightType::NUM_LIGHTS));
			set_light_medium(posLights.memory, static_cast<LightType>(posLights.root.type), scene);
		} else {
			// Determine what level of the tree the light is on
			const u32 level = std::numeric_limits<u32>::digits - 1u - static_cast<u32>(__clz(INTERNAL_NODE_COUNT + lightIdx + 1u));
			// Determine the light's node index within its level
			const u32 levelIndex = (INTERNAL_NODE_COUNT + lightIdx) - ((1u << level) - 1u);
			// The corresponding parent node's level index is then the level index / 2
			const u32 parentLevelIndex = levelIndex / 2u;
			// Finally, compute the tree index of the node
			const u32 parentIndex = (1u << (level - 1u)) - 1u + parentLevelIndex;
			const LightSubTree::Node& node = *posLights.get_node(parentIndex * sizeof(LightSubTree::Node));

			// Left vs. right node
			// TODO: for better divergence let all even indices be processes first, then the uneven ones
			if(levelIndex % 2 == 0) {
				mAssert(node.left.type < static_cast<u16>(LightType::NUM_LIGHTS));
				set_light_medium(&posLights.memory[node.left.offset], static_cast<LightType>(node.left.type), scene);
			} else {
				mAssert(node.right.type < static_cast<u16>(LightType::NUM_LIGHTS));
				set_light_medium(&posLights.memory[node.right.offset], static_cast<LightType>(node.right.type), scene);
			}
		}
	}
#endif // __CUDA_ARCH__
}

namespace lighttree_detail {

void update_media_cuda(const SceneDescriptor<Device::CUDA>& scene, const LightSubTree& posLights) {
	const u32 NODE_COUNT = static_cast<u32>(posLights.internalNodeCount);

	int blockSize = 256;
	int gridSize;
	if(posLights.lightCount == 0)
		gridSize = 1;
	else
		gridSize = (static_cast<int>(posLights.lightCount) - 1) / blockSize + 1;

	cuda_set_medium<<<gridSize, blockSize>>>(scene, posLights, NODE_COUNT,
											 static_cast<u32>(std::log2(posLights.lightCount)));
	cuda::check_error(cudaGetLastError());
}

} // namespace lighttree_detail

}}} // namespace mufflon::scene::lights