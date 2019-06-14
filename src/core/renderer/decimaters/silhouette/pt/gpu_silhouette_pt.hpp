#pragma once

#include "decimation_common_pt.hpp"
#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/allocator.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::silhouette {

template < Device >
struct RenderBuffer;

class GpuShadowSilhouettesPT final : public RendererBase<Device::CUDA> {
public:
	// Initialize all resources required by this renderer.
	GpuShadowSilhouettesPT();
	~GpuShadowSilhouettesPT() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Shadow Silhouette PT"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void pre_reset() final;
	void on_world_clearing() final;
	void post_iteration(OutputHandler& outputBuffer) final;

private:
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void gather_importance();
	void update_reduction_factors();
	void initialize_decimaters();
	void compute_max_importance();
	void display_importance();

	pt::SilhouetteParameters m_params = {};
	math::Rng m_rng;
	std::unique_ptr<u32[]> m_seeds;
	unique_device_ptr<Device::CUDA, u32[]> m_seedsPtr;

	float m_maxImportance = 0.f;
	std::vector<std::unique_ptr<pt::ImportanceDecimater<Device::CUDA>>> m_decimaters;
	// Stores the importance's of each mesh on the GPU/CPU
	unique_device_ptr<Device::CUDA, ArrayDevHandle_t<Device::CUDA, pt::Importances<Device::CUDA>>[]> m_importances;
	unique_device_ptr<Device::CUDA, pt::DeviceImportanceSums<Device::CUDA>[]> m_importanceSums;
	std::vector<double> m_remainingVertexFactor;

	// Superfluous
	u32 m_currentDecimationIteration = 0u;
};

} // namespace mufflon::renderer::decimaters::silhouette