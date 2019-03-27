#pragma once

#include "neb_params.hpp"
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
namespace { using NebPathVertex = PathVertex<struct NebVertexExt>; }

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
		float prevRelativeProbabilitySum;		// Sum of relative probabilities for merges and the connection up to the previous vertex.
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
private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	NebParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	HashGridManager<NebPathVertex> m_viewVertexMapManager;
	HashGrid<Device::CPU, NebPathVertex> m_viewVertexMap;

	HashGridManager<PhotonDesc> m_photonMapManager;
	HashGrid<Device::CPU, PhotonDesc> m_photonMap;
};

} // namespace mufflon::renderer