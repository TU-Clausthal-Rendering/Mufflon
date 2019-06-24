#pragma once

#include "bpm_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/data_structs/photon_map.hpp"
#include "core/data_structs/kdtree.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

template < typename ExtensionT >
class PathVertex;
namespace { using BpmPathVertex = PathVertex<struct BpmVertexExt>; }

class CpuBidirPhotonMapper final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuBidirPhotonMapper();
	~CpuBidirPhotonMapper() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Bidirectional Photon Mapper"; }
	static constexpr StringView get_short_name_static() noexcept { return "BPM"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

	// Information which are stored in the photon map
	struct PhotonDescCommon {
		AreaPdf incidentPdf;
		scene::Direction incident;
		int pathLen;
		Spectrum flux;
		float prevPrevRelativeProbabilitySum;	// Sum of relative probabilities for merges and the connection up to the second previous vertex.
		scene::Direction geoNormal;				// Geometric normal at photon hit point. This is crucial for normal correction.
		float prevConversionFactor;				// 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	};
	struct PhotonDesc : public PhotonDescCommon {
		scene::Point position;
	};

	struct PhotonDescKNN : public PhotonDescCommon {
		float mergeArea;						// Due to KNN search, the mergeArea is different for all points
		int previousIdx;
	};
private:
	void trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius);
	// Create one sample path (PT view path with merges)
	void sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	Spectrum merge(const BpmPathVertex& vertex, const PhotonDescCommon& photon);

	BpmParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	data_structs::HashGridManager<PhotonDesc> m_photonMapManager;
	data_structs::HashGrid<Device::CPU, PhotonDesc> m_photonMap;
	data_structs::KdTree<PhotonDescKNN, 3> m_photonMapKd;
	std::vector<int> m_knnQueryMem;
};

} // namespace mufflon::renderer