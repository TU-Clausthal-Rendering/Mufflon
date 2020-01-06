#include "ss_importance_gathering_pt.hpp"
#include "gpu_ss_sil_pt.hpp"
#include "profiler/Gpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

namespace mufflon::renderer::decimaters::silhouette {

using namespace ss;

namespace gpusssil_detail {

cudaError_t call_kernel_sample(const SilhouetteTargets::template RenderBufferType<Device::CUDA>& outputBuffer,
							   scene::SceneDescriptor<Device::CUDA>* scene,
							   math::Rng* rngs, const SilhouetteParameters& params);
cudaError_t call_kernel_postprocess(const SilhouetteTargets::template RenderBufferType<Device::CUDA>& outputBuffer,
									scene::SceneDescriptor<Device::CUDA>* scene,
									math::Rng* rngs, const SilhouetteParameters& params);

void init_rngs(u32 num, int seed, math::Rng* rngs);

} // namespace gpusssil_detail

void GpuSsSilPT::iterate() {
	copy(&m_sceneDesc->lightTree.posGuide, &m_params.neeUsePositionGuide, sizeof(bool));
	const auto NUM_PIXELS = m_outputBuffer.get_num_pixels();

	gpusssil_detail::call_kernel_sample(m_outputBuffer, m_sceneDesc.get(), m_rngs.get(), m_params);
	gpusssil_detail::call_kernel_postprocess(m_outputBuffer, m_sceneDesc.get(), m_rngs.get(), m_params);
}

void GpuSsSilPT::post_reset() {
	if(!m_rngs || m_rngs.get_deleter().get_size() != static_cast<std::size_t>(m_outputBuffer.get_num_pixels()))
		m_rngs = make_udevptr_array<Device::CUDA, math::Rng, false>(m_outputBuffer.get_num_pixels());
	int seed = 0;// m_params.seed * (m_outputBuffer.get_num_pixels() + 1);
	gpusssil_detail::init_rngs(m_outputBuffer.get_num_pixels(), seed, m_rngs.get());

	// We always account for the background, even if it may be black
	//m_lightCount = 1u + m_sceneDesc.lightTree.posLights.lightCount + m_sceneDesc.lightTree.dirLights.lightCount;
	// TODO: where do we get the light count from?
	const auto statusCount = m_params.maxPathLength * m_lightCount
		* static_cast<std::size_t>(m_outputBuffer.get_num_pixels());
	m_shadowStatus = make_udevptr_array<Device::CUDA, ss::ShadowStatus, false>(statusCount);
	std::memset(m_shadowStatus.get(), 0, sizeof(ss::ShadowStatus) * statusCount);
}

} // namespace mufflon::renderer::decimaters::silhouette