#pragma once

#include "ss_pt_params.hpp"
#include "ss_pt_common.hpp"
#include "ss_decimation_common_pt.hpp"
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

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "SS Silhouette PT"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_reset() final;
	void on_world_clearing() final;
	void post_iteration(IOutputHandler& outputBuffer) final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	void gather_importance();
	void update_reduction_factors();
	void initialize_decimaters();
	void compute_max_importance();
	void display_importance();
	void update_silhouette_importance();

	ss::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	float m_maxImportance = 0.f;
	std::vector<std::unique_ptr<ss::ImportanceDecimater<Device::CPU>>> m_decimaters;
	// Stores the importance's of each mesh on the GPU/CPU (ptr to actual arrays)
	std::vector<ArrayDevHandle_t<Device::CPU, ss::Importances<Device::CPU>>> m_importances;
	std::vector<ss::DeviceImportanceSums<Device::CPU>> m_importanceSums;
	std::vector<ss::SilhouetteEdge> m_shadowPrims;
	std::vector<u32> m_shadowCounts;
	std::vector<double> m_remainingVertexFactor;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
};

} // namespace mufflon::renderer::decimaters::silhouette