#pragma once

#include "decimation_common_pt.hpp"
#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include "core/data_structs/dm_hashgrid.hpp"
#include "core/data_structs/count_octree.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::silhouette {

template < Device >
struct RenderBuffer;

class CpuShadowSilhouettesPT final : public RendererBase<Device::CPU, pt::SilhouetteTargets> {
public:
	// Initialize all resources required by this renderer.
	CpuShadowSilhouettesPT(mufflon::scene::WorldContainer& world);
	~CpuShadowSilhouettesPT() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Shadow Silhouette PT"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_reset() final;
	void on_world_clearing() final;
	void post_iteration(IOutputHandler& outputBuffer) final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	void gather_importance();
	void update_reduction_factors();
	void initialize_decimaters();
	void compute_max_importance();
	void display_importance();

	pt::SilhouetteParameters m_params = {};
	std::vector<math::Rng> m_rngs;

	float m_maxImportance = 0.f;
	std::vector<std::unique_ptr<pt::ImportanceDecimater<Device::CPU>>> m_decimaters;
	// Stores the importance's of each mesh on the GPU/CPU (ptr to actual arrays)
	unique_device_ptr<Device::CPU, ArrayDevHandle_t<Device::CPU, pt::Importances<Device::CPU>>[]> m_importances;
	unique_device_ptr<Device::CPU, pt::DeviceImportanceSums<Device::CPU>[]> m_importanceSums;
	std::vector<std::size_t> m_remainingVertices;

	std::unique_ptr<data_structs::CountOctreeManager> m_viewOctree{};
	std::unique_ptr<data_structs::CountOctreeManager> m_irradianceOctree{};
	std::unique_ptr<data_structs::DmHashGrid<float>> m_viewGrid{};
	std::unique_ptr<data_structs::DmHashGrid<float>> m_irradianceGrid{};
	std::unique_ptr<data_structs::DmHashGrid<u32>> m_irradianceCountGrid{};

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
};

} // namespace mufflon::renderer::decimaters::silhouette
