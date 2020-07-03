#pragma once

#include "neb_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/data_structs/photon_map.hpp"
#include "core/data_structs/dm_octree.hpp"
#include "core/data_structs/kdtree.hpp"
#include <vector>

//#define NEB_KDTREE

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

struct NebVertexExt {
	Spectrum throughput;
	AngularPdf pdfBack;
	AreaPdf incidentPdf{ 0.0f };
	union {
		int pixelIndex;
		float rnd;		// An additional input random value (first vertex of light paths only).
	};
	float density{ -1.0f };

	CUDA_FUNCTION void init(const PathVertex<NebVertexExt>& /*thisVertex*/,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice);

	CUDA_FUNCTION void update(const PathVertex<NebVertexExt>& prevVertex,
							  const PathVertex<NebVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float /*continuationPropability*/,
							  const Spectrum& /*transmission*/);

	CUDA_FUNCTION void update(const PathVertex<NebVertexExt>& /*thisVertex*/,
							  const scene::Direction& /*excident*/,
							  const VertexSample& sample);
};

class NebPathVertex : public PathVertex<NebVertexExt> {
public:
	// Overload the vertex sample operator to have more RR control.
	CUDA_FUNCTION VertexSample sample(const ei::Box& sceneBounds,
									  const scene::materials::Medium* media,
									  const math::RndSet2_1& rndSet,
									  bool adjoint) const;
};

class CpuNextEventBacktracking final : public RendererBase<Device::CPU, NebTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuNextEventBacktracking(mufflon::scene::WorldContainer& world) :
		RendererBase<Device::CPU, NebTargets>{ world }
	{}
	~CpuNextEventBacktracking() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Next Event Backtracking"; }
	static constexpr StringView get_short_name_static() noexcept { return "NEB"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

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
		const NebPathVertex* previous;	// The previous vertex to compute the reuseCount after the density estimate
		Spectrum radiance;				// emission.value * throughput
		AreaPdf incidentPdf;			// From random hit
		AreaPdf startPdf;				// connectPdf
		AngularPdf samplePdf;			// samplePdf * emitPdf / connectPdf
		scene::Direction incident;
		float incidentDistSq;
		float maxIrradiance;			// max(emission.value) * cosθprev / (connectPdf * d²)
	};

	struct NeeDesc {
		Spectrum irradiance;
		scene::Direction direction;
		float conversion;		// Partial evaluation of the relPdf for the next event: (cosθ / d²)
		AreaPdf creationPdf;
		AreaPdf backPdf;		// neeSamplePdf * cosθs / d²
		float distance;
	};
private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	//EmissionDesc evaluate_self_radiance(const NebPathVertex& vertex, bool includeThroughput);
	//EmissionDesc evaluate_background(const NebPathVertex& vertex, const VertexSample& sample, int pathLen);
	void sample_view_path(const Pixel coord, const int pixelIdx);
	void sample_photon_path(float photonMergeArea, math::Rng& rng, const NebPathVertex& vertex, const NeeDesc& nee, AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons);
	void sample_std_photon(int idx, int numPhotons, u64 seed, float photonMergeArea, AreaPdf* incidentF, AreaPdf* incidentB);
	void merge_importons(float photonMergeArea, const NebPathVertex& lvertex, AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons, bool isStd, const Spectrum& flux, int pdfHead, float prevConversionFactor, float sourceDensity);
	NeeDesc compute_nee(math::Rng& rng, const NebPathVertex& vertex);
	void evaluate_nee(const NebPathVertex& vertex, const NeeDesc& nee, AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons, float photonMergeArea);
	Spectrum finalize_emission(float photonMergeArea, const EmissionDesc& emission, AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons);

	NebParameters m_params = {};
	float m_targetFlux;
	std::vector<math::Rng> m_rngs;
	data_structs::HashGridManager<NebPathVertex> m_viewVertexMapManager;
	data_structs::HashGrid<Device::CPU, NebPathVertex> m_viewVertexMap;
	std::vector<EmissionDesc> m_selfEmissiveEndVertices;
	std::atomic_int32_t m_selfEmissionCount;
#ifdef NEB_KDTREE
	data_structs::KdTree<char, 3> m_density;		// A kd-tree with positions only, TODO: data is not needed
#else
	std::unique_ptr<data_structs::DmOctree<>> m_density;
#endif
};

} // namespace mufflon::renderer