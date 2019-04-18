#pragma once

#include "cpu_silhouette_decimater.hpp"
#include "core/renderer/decimaters/silhouette/silhouette_params.hpp"
#include "core/renderer/decimaters/silhouette/sil_common.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters {

template < Device >
struct RenderBuffer;

class CpuShadowSilhouettes final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettes();
	~CpuShadowSilhouettes() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Shadow Silhouette"; }
	StringView get_short_name() const noexcept final { return "SS"; }

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
	bool trace_shadow_silhouette(const ei::Ray& shadowRay, const silhouette::SilPathVertex& vertex,
								 const float importance);
	bool trace_shadow(const ei::Ray& shadowRay, const silhouette::SilPathVertex& vertex,
					  const float importance);

	u32 get_memory_requirement() const;

	void update_reduction_factors();

	void initialize_decimaters();

	silhouette::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	std::vector<std::unique_ptr<silhouette::CpuImportanceDecimater>> m_decimaters;
	std::vector<double> m_remainingVertexFactor;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
	float m_maxImportance;
};

} // namespace mufflon::renderer::decimaters