#pragma once

#include "shadow_photons_params.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/data_structs/dm_octree.hpp"
#include "core/data_structs/dm_hashgrid.hpp"
#include <ei/vector.hpp>
#include <optional>
#include <vector>

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
	void display_photon_densities(const ei::IVec2& coord, const SpvPathVertex& vertex);

	void trace_photon(const int idx, const int numPhotons, const u64 seed);
	float query_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
							   ei::Vec3* gradient = nullptr) const;
	float query_shadow_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
									  ei::Vec3* gradient = nullptr) const;
	void init_rngs(const int num);

	std::optional<ei::Vec3> trace_shadow_photon(const SpvPathVertex& vertex, const int idx);

	ShadowPhotonParameters m_params;
	std::vector<math::Rng> m_rngs;

	// TODO: unify this in one data structure (possibly by storing which light a photon came from alongside count?)


	std::vector<data_structs::DmOctree> m_densityPhotonsOctree;
	std::vector<data_structs::DmOctree> m_densityShadowPhotonsOctree;
	std::vector<data_structs::DmHashGrid> m_densityPhotonsHashgrid;
	std::vector<data_structs::DmHashGrid> m_densityShadowPhotonsHashgrid;
};

} // namespace mufflon::renderer::decimaters::spm