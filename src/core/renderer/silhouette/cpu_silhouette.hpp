#pragma once

#include "silhouette_params.hpp"
#include "sil_common.hpp"
#include "decimation/importance_decimater.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuShadowSilhouettes final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettes();
	~CpuShadowSilhouettes() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Shadow silhouettes"; }
	StringView get_short_name() const noexcept final { return "SS"; }

	void pre_descriptor_requery() final;
	bool pre_iteration(OutputHandler& outputBuffer) final;
	void post_iteration(OutputHandler& outputBuffer) final;
	void on_scene_load() final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	void importance_sample(const Pixel coord);
	void pt_sample(const Pixel coord);

	void gather_importance();
	void compute_max_importance();
	void display_importance();
	float query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId);
	bool trace_shadow_silhouette(const ei::Ray& shadowRay, const silhouette::SilPathVertex& vertex,
								 const float importance);

	u32 get_memory_requirement() const;

	void initialize_decimaters();

	SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	std::vector<silhouette::decimation::ImportanceDecimater> m_decimaters;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
	float m_maxImportance;
};

} // namespace mufflon::renderer