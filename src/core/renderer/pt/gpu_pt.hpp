#pragma once

#include "pt_params.hpp"
#include "core/memory/allocator.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

template < Device >
struct RenderBuffer;

class GpuPathTracer : public RendererBase<Device::CUDA> {
public:
	GpuPathTracer();
	~GpuPathTracer() = default;

	// This is just a test method, don't use this as an actual interface
	void iterate() override;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept { return "Pathtracer"; }
	StringView get_short_name() const noexcept final { return "PT"; }

	void on_descriptor_requery() final;

private:
	PtParameters m_params;
	math::Rng m_rng;
	std::unique_ptr<u32[]> m_seeds;
	unique_device_ptr<Device::CUDA, u32[]> m_seedsPtr;
};

}} // namespace mufflon::renderer