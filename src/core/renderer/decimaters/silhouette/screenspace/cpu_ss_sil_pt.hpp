#pragma once

#include "ss_pt_params.hpp"
#include "ss_pt_common.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::silhouette {

template < Device >
struct RenderBuffer;

class CpuSsSilPT final : public RendererBase<Device::CPU, ss::SilhouetteTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuSsSilPT();
	~CpuSsSilPT() = default;

	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "SS Silhouette PT"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void iterate() final;
	void post_reset() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	ss::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	// Stores the importance's of each mesh on the GPU/CPU (ptr to actual arrays)
	//std::vector<ArrayDevHandle_t<Device::CPU, ss::Importances<Device::CPU>>> m_importances;
	//std::vector<ss::DeviceImportanceSums<Device::CPU>> m_importanceSums;
	//std::vector<ss::SilhouetteEdge> m_shadowPrims;
	//std::vector<u8> m_shadowStatus;	// 2 bits per light: 00 - not tested; 01 - shadowed; 10 - lit; 11 - both
	std::vector<ss::ShadowStatus> m_shadowStatus;
	std::size_t m_lightCount = 0u;
	std::size_t m_bytesPerPixel = 0u;
	std::vector<double> m_remainingVertexFactor;
};

} // namespace mufflon::renderer::decimaters::silhouette