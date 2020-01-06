#pragma once

#include "vcm_params.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/data_structs/photon_map.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < typename ExtensionT >
class PathVertex;
using VcmPathVertex = PathVertex<struct VcmVertexExt>;

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct VcmVertexExt {
	AreaPdf incidentPdf;
	Spectrum throughput;
	// A cache to shorten the recursive evaluation of MIS (relative to connections).
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// Only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor{ 0.0f };

	CUDA_FUNCTION void init(const VcmPathVertex& /*thisVertex*/,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice);

	CUDA_FUNCTION void update(const VcmPathVertex& prevVertex,
							  const VcmPathVertex& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float/* continuationPropability*/,
							  const Spectrum& /*transmission*/,
							  int /*numPhotons*/, float /*area*/);

	inline CUDA_FUNCTION void update(const VcmPathVertex& thisVertex,
									 const scene::Direction& /*excident*/,
									 const VertexSample& sample,
									 int numPhotons, float area);
};

class CpuVcm final : public RendererBase<Device::CPU, VcmTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuVcm(mufflon::scene::WorldContainer& world) :
		RendererBase<Device::CPU, VcmTargets>{ world }
	{}
	~CpuVcm() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Vertex Connection and Merging"; }
	static constexpr StringView get_short_name_static() noexcept { return "VCM"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

	// Information which are stored in the photon map
	/*struct PhotonDesc {
		scene::Point position;
		AreaPdf incidentPdf;
		scene::Direction incident;
		int pathLen;
		Spectrum flux;
		float prevPrevRelativeProbabilitySum;	// Sum of relative probabilities for merges and the connection up to the second previous vertex.
		scene::Direction geoNormal;				// Geometric normal at photon hit point. This is crucial for normal correction.
		float prevConversionFactor;				// 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	};*/
private:
	void trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius);
	// Create one sample path (PT view path with merges)
	void sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	VcmParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	data_structs::HashGridManager<VcmPathVertex> m_photonMapManager;
	data_structs::HashGrid<Device::CPU, VcmPathVertex> m_photonMap;
	std::vector<const VcmPathVertex*> m_pathEndPoints;
};

} // namespace mufflon::renderer