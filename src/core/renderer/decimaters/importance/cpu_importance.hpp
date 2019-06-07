#pragma once

#include "importance_params.hpp"
#include "imp_common.hpp"
#include "importance_decimater.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters {

template < Device >
struct RenderBuffer;

class CpuImportanceDecimater final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuImportanceDecimater();
	~CpuImportanceDecimater() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Importance decimater"; }
	static constexpr StringView get_short_name_static() noexcept { return "ImpD"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_descriptor_requery() final;
	void post_iteration(OutputHandler& outputBuffer) final;
	void on_scene_load() final;
	void on_scene_unload() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	void importance_sample(const Pixel coord);
	void pt_sample(const Pixel coord);

	void gather_importance();
	void compute_max_importance();
	float query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId);

	u32 get_memory_requirement() const;

	void update_reduction_factors();

	void initialize_decimaters();

	importance::ImportanceParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	std::vector<std::unique_ptr<importance::ImportanceDecimater>> m_decimaters;
	std::vector<double> m_remainingVertexFactor;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
	float m_maxImportance;
};

} // namespace mufflon::renderer::decimaters