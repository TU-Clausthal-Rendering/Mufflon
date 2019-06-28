#pragma once

#include "shadow_photons_params.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/data_structs/dm_octree.hpp"
#include <ei/vector.hpp>
#include <optional>

namespace mufflon::renderer::decimaters::spm {

struct SpvVertexExt {
	CUDA_FUNCTION void init(const PathVertex<SpvVertexExt>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosineAbs,
							const math::Throughput& incidentThrougput) {
	}

	CUDA_FUNCTION void update(const PathVertex<SpvVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
	}
};
using SpvPathVertex = PathVertex<SpvVertexExt>;

class ShadowPhotonVisualizer final : public RendererBase<Device::CPU> {
public:
	ShadowPhotonVisualizer();
	~ShadowPhotonVisualizer();

	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Shadow Photon Visualizer"; }
	static constexpr StringView get_short_name_static() noexcept { return "SPV"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void iterate() final;
	void post_reset() final;

private:
	void trace_photon(const int idx, const int numPhotons, const u64 seed);
	float query_photon_density(const SpvPathVertex& vertex, ei::Vec3* gradient = nullptr);
	float query_shadow_photon_density(const SpvPathVertex& vertex, ei::Vec3* gradient = nullptr);
	void init_rngs(const int num);

	std::optional<ei::Vec3> trace_shadow_photon(const SpvPathVertex& vertex, const int idx);

	ShadowPhotonParameters m_params;
	std::vector<math::Rng> m_rngs;
	std::unique_ptr<data_structs::DmOctree> m_densityPhotons;
	std::unique_ptr<data_structs::DmOctree> m_densityShadowPhotons;
};

} // namespace mufflon::renderer::decimaters::spm