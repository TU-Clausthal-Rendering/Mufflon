#pragma once

#include "lt_params.hpp"
#include "core/memory/allocator.hpp"
#include "core/renderer/renderer_base.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

template < Device >
struct RenderBuffer;

class GpuLightTracer final : public RendererBase<Device::CUDA> {
public:
	GpuLightTracer();
	~GpuLightTracer() = default;

	// This is just a test method, don't use this as an actual interface
	void iterate() override;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Lighttracer"; }
	StringView get_short_name() const noexcept final { return "LT"; }
	void on_reset() final;
	void on_descriptor_requery() final;

private:
	LtParameters m_params;
	unique_device_ptr<Device::CUDA, math::Rng[]> m_rngs;
};

}} // namespace mufflon::renderer