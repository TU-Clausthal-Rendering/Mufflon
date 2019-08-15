#pragma once

#include "pt_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/allocator.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

class GpuPathTracer final : public RendererBase<Device::CUDA, PtTargets> {
public:
	GpuPathTracer();
	~GpuPathTracer() = default;

	// This is just a test method, don't use this as an actual interface
	void iterate() override;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Pathtracer"; }
	static constexpr StringView get_short_name_static() noexcept { return "PT"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }
	void post_reset() final;

private:
	PtParameters m_params;
	unique_device_ptr<Device::CUDA, math::Rng[]> m_rngs;
};

}} // namespace mufflon::renderer
