#pragma once

#include "wireframe_params.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

template < Device >
struct RenderBuffer;

class GpuWireframe : public RendererBase<Device::CUDA> {
public:
	GpuWireframe();
	~GpuWireframe() = default;

	// This is just a test method, don't use this as an actual interface
	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Wireframe"; }
	StringView get_short_name() const noexcept final { return "WF"; }

	void on_descriptor_requery() final;

private:
	WireframeParameters m_params = {};
	math::Rng m_rng;
	std::unique_ptr<u32[]> m_seeds;
	unique_device_ptr<Device::CUDA, u32[]> m_seedsPtr;
};

}}// namespace mufflon::renderer