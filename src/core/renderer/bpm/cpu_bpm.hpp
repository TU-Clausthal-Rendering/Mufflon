#pragma once

#include "bpm_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/photon_map.hpp"
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
	StringView get_name() const noexcept final { return "Bidirectional Photon Mapper"; }
	StringView get_short_name() const noexcept final { return "BPM"; }

	void on_reset() final;

private:
	void trace_photon(int idx, int numPhotons, u64 seed);
	// Create one sample path (PT view path with merges)
	void sample(const Pixel coord, int idx, float currentMergeRadius);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	struct PhotonDesc {
		scene::Point position;
		AreaPdf incidentPdf;
		scene::Direction incident;
		int pathLen;
		Spectrum flux;
		scene::Direction geoNormal;
	};

	BpmParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	HashGridManager<PhotonDesc> m_photonMapManager;
	HashGrid<Device::CPU, PhotonDesc> m_photonMap;
};

} // namespace mufflon::renderer