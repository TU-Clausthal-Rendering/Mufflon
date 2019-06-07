#pragma once

#include "renderer.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"
#include "output_handler.hpp"
#include <type_traits>

namespace mufflon::renderer {

template < Device dev >
class RendererBase : public IRenderer {
public:
	static constexpr Device DEVICE = dev;

	RendererBase();

	bool uses_device(Device device) const noexcept override { return may_use_device(device); }
	static constexpr bool may_use_device(Device device) noexcept { return DEVICE == device; }

	bool pre_iteration(OutputHandler& outputBuffer) override;
	void post_iteration(OutputHandler& outputBuffer) override;

protected:
	RenderBuffer<DEVICE> m_outputBuffer;
	OutputValue m_outputTargets;

	// CPU gets the descriptor directly, everyone else gets a unique_ptr
	std::conditional_t<DEVICE == Device::CPU || DEVICE == Device::OPENGL, scene::SceneDescriptor<DEVICE>,
		unique_device_ptr<DEVICE, scene::SceneDescriptor<DEVICE>>> m_sceneDesc;
};

} // namespace mufflon::renderer