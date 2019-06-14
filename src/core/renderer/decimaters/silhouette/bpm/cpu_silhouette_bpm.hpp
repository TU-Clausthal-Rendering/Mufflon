#pragma once

#include "decimation_common_bpm.hpp"
#include "silhouette_bpm_common.hpp"
#include "silhouette_bpm_params.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/renderer/photon_map.hpp"
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::silhouette {

template < Device >
struct RenderBuffer;

class CpuShadowSilhouettesBPM final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettesBPM();
	~CpuShadowSilhouettesBPM() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Shadow Silhouette BPM"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSBPM"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void on_world_clearing() final;
	void pre_reset() final;
	void post_iteration(OutputHandler& outputBuffer) final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	void gather_importance();
	void update_reduction_factors();
	void initialize_decimaters();
	void compute_max_importance();
	void display_importance();

	bpm::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	HashGridManager<bpm::PhotonDesc> m_photonMapManager;
	HashGrid<Device::CPU, bpm::PhotonDesc> m_photonMap;

	float m_maxImportance = 0.f;
	std::vector<std::unique_ptr<bpm::ImportanceDecimater<Device::CPU>>> m_decimaters;
	// Stores the importance's of each mesh on the GPU/CPU (ptr to actual arrays)
	unique_device_ptr<Device::CPU, ArrayDevHandle_t<Device::CPU, bpm::Importances<Device::CPU>>[]> m_importances;
	unique_device_ptr<Device::CPU, bpm::DeviceImportanceSums<Device::CPU>[]> m_importanceSums;
	std::vector<double> m_remainingVertexFactor;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
};

} // namespace mufflon::renderer::decimaters::silhouette