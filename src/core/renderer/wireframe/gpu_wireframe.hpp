#pragma once

#include "wireframe_params.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

class GpuWireframe final : public RendererBase<Device::CUDA, WireframeTargets> {
public:
	GpuWireframe(mufflon::scene::WorldContainer& world);
	~GpuWireframe() = default;

	// This is just a test method, don't use this as an actual interface
	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	static constexpr StringView get_name_static() noexcept { return "Wireframe"; }
	static constexpr StringView get_short_name_static() noexcept { return "WF"; }
	StringView get_name() const noexcept final { return get_name_static(); }
	StringView get_short_name() const noexcept final { return get_short_name_static(); }

	void post_reset() final;

private:
	WireframeParameters m_params = {};
	math::Rng m_rng;
	std::unique_ptr<u32[]> m_seeds;
	unique_device_ptr<Device::CUDA, u32[]> m_seedsPtr;
};

}}// namespace mufflon::renderer
