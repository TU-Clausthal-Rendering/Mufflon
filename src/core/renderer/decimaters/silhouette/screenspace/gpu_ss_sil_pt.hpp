#pragma once

#include "ss_pt_params.hpp"
#include "ss_pt_common.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

namespace mufflon::renderer::decimaters::silhouette {

template < Device >
struct RenderBuffer;

class GpuSsSilPT final : public RendererBase<Device::CUDA, ss::SilhouetteTargets> {
public:
	// Initialize all resources required by this renderer.
	GpuSsSilPT(mufflon::scene::WorldContainer& world) :
		RendererBase<Device::CUDA, ss::SilhouetteTargets>{ world }
	{}
	~GpuSsSilPT() = default;

	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "SS Silhouette PT"; }
	static constexpr StringView get_short_name_static() noexcept { return "SSSPT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void iterate() final;
	void post_reset() final;

private:
	ss::SilhouetteParameters m_params = {};
	unique_device_ptr<Device::CUDA, math::Rng[]> m_rngs;

	unique_device_ptr<Device::CUDA, ss::ShadowStatus[]> m_shadowStatus;
	std::size_t m_lightCount = 0u;
	std::size_t m_bytesPerPixel = 0u;
};

} // namespace mufflon::renderer::decimaters::silhouette