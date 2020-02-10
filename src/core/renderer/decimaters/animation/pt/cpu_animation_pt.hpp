#pragma once

#include "animation_decimation_common_pt.hpp"
#include "animation_pt_params.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/renderer/decimaters/silhouette/pt/silhouette_pt_common.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::animation {

template < Device >
struct RenderBuffer;

class CpuShadowSilhouettesPT final : public RendererBase<Device::CPU, silhouette::pt::SilhouetteTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettesPT(mufflon::scene::WorldContainer& world);
	~CpuShadowSilhouettesPT() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Animated SSPT"; }
	static constexpr StringView get_short_name_static() noexcept { return "ASSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_reset() final;
	void on_world_clearing() final;
	void on_animation_frame_changed(const u32 from, const u32 to) final;
	void post_iteration(IOutputHandler& outputBuffer) final;

private:
	struct PerFrameData {
		// Stores the importance's of each mesh on the GPU/CPU (ptr to actual arrays) and per animation frame
		unique_device_ptr<Device::CPU, silhouette::pt::DeviceImportanceSums<Device::CPU>[]> importanceSums = nullptr;
		float maxImportance = 0.f;
	};

	enum class Stage {
		NONE,
		INITIALIZED,
		IMPORTANCE_GATHERED
	};

	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	void gather_importance();
	void update_reduction_factors();
	void initialize_decimaters();
	void compute_max_importance();
	void display_importance(const bool accumulated = false);

	pt::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	unique_device_ptr<Device::CPU, silhouette::pt::Importances<Device::CPU>[]> m_importanceBuffer;
	unique_device_ptr<Device::CPU, silhouette::pt::Importances<Device::CPU>[]> m_accumImportanceBuffer;
	std::vector<std::unique_ptr<pt::ImportanceDecimater>> m_decimaters;
	unique_device_ptr<Device::CPU, ArrayDevHandle_t<Device::CPU, silhouette::pt::Importances<Device::CPU>>[]> m_importances;
	unique_device_ptr<Device::CPU, ArrayDevHandle_t<Device::CPU, silhouette::pt::Importances<Device::CPU>>[]> m_accumImportances;
	std::vector<PerFrameData> m_perFrameData;
	std::vector<double> m_remainingVertexFactor;

	Stage m_stage = Stage::NONE;
	std::vector<bool> m_reduced;
};

} // namespace mufflon::renderer::decimaters::animation