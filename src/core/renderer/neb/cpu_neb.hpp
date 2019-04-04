#pragma once

#include "neb_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/photon_map.hpp"
#include "density_octree.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class NebPathVertex;
struct NebVertexExt;

class CpuNextEventBacktracking final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuNextEventBacktracking();
	~CpuNextEventBacktracking() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Next Event Backtracking"; }
	StringView get_short_name() const noexcept final { return "NEB"; }

	void on_reset() final;

	// Information which are stored in the photon map
	struct PhotonDesc {
		scene::Point position;
		AreaPdf incidentPdf;
		scene::Direction incident;
		int pathLen;
		Spectrum irradiance;
		float prevPrevRelativeProbabilitySum;	// Sum of relative probabilities for merges and the connection up to the previous vertex.
		scene::Direction geoNormal;				// Geometric normal at photon hit point. This is crucial for normal correction.
		float prevConversionFactor;				// 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	};

	/*struct ImportonDesc {
		ImportonDesc* previous;
		scene::PrimitiveHandle primitive;
		Spectrum throughput;
		i16 pathLen;
		i16 neeCount;
		scene::TangentSpace tangentSpace;
		ei::Vec2 uv;
	};*/

	struct EmissionDesc {
		const NebVertexExt* previous;	// The previous vertex to compute the reuseCount after the density estimate
		Spectrum radiance;				// emission.value
		float relPdf;					// Part of the MIS-weight relative to the NEE event (without the reuseCount)
	};
private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	EmissionDesc evaluate_self_radiance(const NebPathVertex& vertex, bool includeThroughput);
	EmissionDesc evaluate_background(const NebPathVertex& vertex, const VertexSample& sample, int pathLen);
	void sample_view_path(const Pixel coord, const int pixelIdx);
	void estimate_density(float densityEstimateRadiusSq, NebPathVertex& vertex);
	void sample_photon_path(float neeMergeArea, float photonMergeArea, math::Rng& rng, const NebPathVertex& vertex);
	Spectrum merge_photons(float mergeRadiusSq, const NebPathVertex& vertex);
	Spectrum evaluate_nee(const NebPathVertex& vertex, const NebVertexExt& ext, float neeReuseCount, float mergeReuseCount);
	Spectrum merge_nees(float mergeRadiusSq, float photonMergeArea, const NebPathVertex& vertex);
	Spectrum finalize_emission(float neeMergeArea, float photonMergeArea, const EmissionDesc& emission);

	NebParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	HashGridManager<NebPathVertex> m_viewVertexMapManager;
	HashGrid<Device::CPU, NebPathVertex> m_viewVertexMap;
	std::vector<EmissionDesc> m_selfEmissiveEndVertices;
	std::atomic_int32_t m_selfEmissionCount;
	HashGridManager<PhotonDesc> m_photonMapManager;
	HashGrid<Device::CPU, PhotonDesc> m_photonMap;
	DensityOctree m_density;
};

} // namespace mufflon::renderer