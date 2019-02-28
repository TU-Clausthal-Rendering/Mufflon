#pragma once

#include "silhouette_params.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/renderer/importance/importance_map.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

template < typename T, int A >
class PathVertex;

class CpuShadowSilhouettes final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettes();
	~CpuShadowSilhouettes() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Shadow silhouettes"; }
	StringView get_short_name() const noexcept final { return "SS"; }

	void on_descriptor_requery() final;
	bool pre_iteration(OutputHandler& outputBuffer) final;
	void on_scene_load() final;

private:
	using PtPathVertex = PathVertex<u8, 4>;

	// Create one sample path (actual PT algorithm)
	void pt_sample(const Pixel coord);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	void importance_sample(const Pixel coord);

	void initialize_importance_map();
	void gather_importance();
	bool trace_shadow_silhouette(const ei::Ray& shadowRay, const PtPathVertex& vertex,
								 const scene::PrimitiveHandle& firstHit,
								 const float lightDist, const float firstHitT,
								 const float importance);
	void decimate();
	void undecimate();
	void compute_max_importance();
	void display_importance();
	float query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId);

	SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	ImportanceMap m_importanceMap;

	// Superfluous
	bool m_addedLods = false;
	bool m_finishedDecimation = false;
	u32 m_currentDecimationIteration = 0u;
	float m_maxImportance;
};

} // namespace mufflon::renderer