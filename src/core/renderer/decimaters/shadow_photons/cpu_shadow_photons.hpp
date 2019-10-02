#pragma once

#include "shadow_photons_params.hpp"
#include "core/export/api.h"
#include "core/renderer/path_util.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/data_structs/dm_octree.hpp"
#include "core/data_structs/dm_hashgrid.hpp"
#include "core/data_structs/photon_map.hpp"
#include <ei/vector.hpp>
#include <optional>
#include <vector>

namespace mufflon::renderer::decimaters::spm {

struct SpvVertexExt {
	CUDA_FUNCTION void init(const PathVertex<SpvVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {}

	CUDA_FUNCTION void update(const PathVertex<SpvVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const VertexSample& sample) {}

	CUDA_FUNCTION void update(const PathVertex<SpvVertexExt>& prevVertex,
							  const PathVertex<SpvVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float continuationPropability,
							  const Spectrum& transmission) {}
};
using SpvPathVertex = PathVertex<SpvVertexExt>;

class ShadowPhotonVisualizer final : public RendererBase<Device::CPU, ShadowPhotonTargets> {
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
	// Information which are stored in the photon map
	struct PhotonDesc {
		scene::Direction incident;
		int pathLen;
		Spectrum flux;
		scene::Direction geoNormal;				// Geometric normal at photon hit point. This is crucial for normal correction.
		ei::Vec3 position;
		ei::Vec3 closestVertexPos;
		PhotonDesc* prevPhoton;
	};

	void display_photon_densities(const ei::IVec2& coord, const SpvPathVertex& vertex);
	void display_silhouette(const ei::IVec2& coord, const i32 index, const SpvPathVertex& vertex);

	void trace_photon(const int idx, const int numPhotons, const u64 seed);
	bool trace_shadow_silhouette(const ei::Ray& shadowRay, const float shadowDistance,
								 const float lightDistance, const scene::PrimitiveHandle& shadowHit,
								 const SpvPathVertex& vertex) const;
	std::optional<ei::Vec3> trace_shadow_photon(const SpvPathVertex& vertex, const int idx);

	float query_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
							   ei::Vec3* gradient = nullptr) const;
	float query_shadow_photon_density(const SpvPathVertex& vertex, const std::size_t lightIndex,
									  ei::Vec3* gradient = nullptr) const;

	std::array<ei::Vec3, 3u> get_triangle_vertices(const scene::PrimitiveHandle& hitId) const;
	std::array<ei::Vec3, 4u> get_quad_vertices(const scene::PrimitiveHandle& hitId) const;
	std::optional<ei::Vec3> get_closest_vertex(const ei::Vec3& hitpoint,
											   const scene::PrimitiveHandle& hitId) const;

	void init_rngs(const int num);

	ShadowPhotonParameters m_params;
	std::vector<math::Rng> m_rngs;

	// TODO: unify this in one data structure (possibly by storing which light a photon came from alongside count?)
	std::vector<data_structs::DmOctree<>> m_densityPhotonsOctree;
	std::vector<data_structs::DmOctree<>> m_densityShadowPhotonsOctree;
	std::vector<data_structs::DmHashGrid> m_densityPhotonsHashgrid;
	std::vector<data_structs::DmHashGrid> m_densityShadowPhotonsHashgrid;

	data_structs::HashGridManager<PhotonDesc> m_photonMapManager;
	data_structs::HashGrid<Device::CPU, PhotonDesc> m_photonMap;

	std::unique_ptr<data_structs::DmOctree<float>> m_importance;
};

} // namespace mufflon::renderer::decimaters::spm