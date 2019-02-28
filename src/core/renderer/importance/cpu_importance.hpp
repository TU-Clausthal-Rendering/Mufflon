#pragma once

#include "importance_params.hpp"
#include "importance_map.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

// Decimation according to 'Illumination-driven Mesh Reduction for Accelerating Light Transport Simulations' (Andreas Reich, 2015)

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuImportanceDecimater final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuImportanceDecimater();
	~CpuImportanceDecimater() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Importance decimation"; }
	StringView get_short_name() const noexcept final { return "ImpD"; }

	void on_descriptor_requery() final;

private:
	// Create one sample path (actual PT algorithm)
	void pt_sample(const Pixel coord);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	void importance_sample(const Pixel coord);

	void initialize_importance_map();
	void gather_importance();
	void decimate();
	void compute_max_importance();
	void display_importance();
	float query_importance(const ei::Vec3& hitPoint, const scene::PrimitiveHandle& hitId);

	ImportanceParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	// Data buffer for importance
	ImportanceMap m_importanceMap;

	// Superfluous
	bool m_gotImportance = false;
	bool m_finishedDecimation = false;
	int m_currentImportanceIteration = 0;
	float m_maxImportance;
};

} // namespace mufflon::renderer