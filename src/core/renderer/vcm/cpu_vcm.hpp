#pragma once

#include "vcm_params.hpp"
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
namespace { using VcmPathVertex = PathVertex<struct VcmVertexExt>; }

class CpuVcm final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuVcm();
	~CpuVcm();

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Vertex Connection and Merging"; }
	static constexpr StringView get_short_name_static() noexcept { return "VCM"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void on_reset() final;

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
	HashGridManager<VcmPathVertex> m_photonMapManager;
	HashGrid<Device::CPU, VcmPathVertex> m_photonMap;
	std::vector<const VcmPathVertex*> m_pathEndPoints;
};

} // namespace mufflon::renderer