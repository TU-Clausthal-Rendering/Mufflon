#pragma once

#include "ivcm_params.hpp"
#include "core/scene/handles.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/data_structs/photon_map.hpp"
#include "core/data_structs/dm_hashgrid.hpp"
#include "core/data_structs/dm_octree.hpp"
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
namespace { using IvcmPathVertex = PathVertex<struct IvcmVertexExt>; }

class CpuIvcm final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuIvcm();
	~CpuIvcm();

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Improved Vertex Connection and Merging"; }
	static constexpr StringView get_short_name_static() noexcept { return "IVCM"; }
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
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal, float currentMergeRadius) const;
	void trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius);
	// Create one sample path (PT view path with merges)
	void sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius,
				AreaPdf* incidentF, AreaPdf* incidentB, IvcmPathVertex* vertexBuffer,
				int* reuseCount);
	void compute_counts(int* reuseCount, float mergeArea, int numPhotons,
						const IvcmPathVertex* path0, int pl0,
						const IvcmPathVertex* path1, int pl1);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	IvcmParameters m_params = {};
	std::vector<math::Rng> m_rngs;
	data_structs::HashGridManager<IvcmPathVertex> m_photonMapManager;
	data_structs::HashGrid<Device::CPU, IvcmPathVertex> m_photonMap;
	std::vector<const IvcmPathVertex*> m_pathEndPoints;
	std::vector<AreaPdf> m_tmpPathProbabilities;
	std::vector<int> m_tmpReuseCounts;
	std::vector<IvcmPathVertex> m_tmpViewPathVertices;
	std::unique_ptr<data_structs::DmOctree> m_density;
	//std::unique_ptr<data_structs::DmHashGrid> m_density;
	std::unique_ptr<data_structs::KdTree<char,3>> m_density2;
};

} // namespace mufflon::renderer