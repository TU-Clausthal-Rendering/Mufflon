#pragma once

#include "combined_common.hpp"
#include "combined_params.hpp"
#include "util/swapped_vector.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/renderer/decimaters/util/octree.hpp"
#include "core/renderer/decimaters/util/octree_manager.hpp"

namespace mufflon::renderer::decimaters {

class CpuCombinedReducer final : public RendererBase<Device::CPU, combined::CombinedTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuCombinedReducer(mufflon::scene::WorldContainer& world);
	~CpuCombinedReducer() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Combined Reducer"; }
	static constexpr StringView get_short_name_static() noexcept { return "CR"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_reset() final;
	void post_reset() final;
	void on_world_clearing() final;
	void post_iteration(IOutputHandler& outputBuffer) final;

private:
	enum class Stage {
		NONE,
		INITIALIZED,
		IMPORTANCE_GATHERED
	};

	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	void gather_importance();
	void update_reduction_factors(u32 frameStart, u32 frameEnd);
	void initialize_decimaters();
#if 0
	void display_importance(const bool accumulated = false);
#endif // 0
	double get_lod_importance(const u32 frame, const scene::Scene::InstanceRef obj) const noexcept;

	combined::CombinedParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	float m_maxImportance = 0.f;
	std::vector<std::size_t> m_originalVertices;
	std::vector<std::size_t> m_remainingVertices;

	std::vector<OctreeManager<FloatOctree>> m_viewOctrees;
	std::vector<OctreeManager<SampleOctree>> m_irradianceOctrees;
	std::unique_ptr<util::SwappedVector<std::atomic<double>>> m_instanceImportanceSums;
	// Shadow screenspace info
	unique_device_ptr<Device::CPU, combined::ShadowStatus[]> m_shadowStatus;
	std::size_t m_lightCount = 0u;

	// Debug
	std::vector<float> m_accumImportances;

	// Superfluous
	Stage m_stage = Stage::NONE;
	std::vector<bool> m_hasFrameImp;
	std::vector<bool> m_isFrameReduced;


};

} // namespace mufflon::renderer::decimaters