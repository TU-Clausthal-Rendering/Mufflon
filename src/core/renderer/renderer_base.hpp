#pragma once

#include "renderer.hpp"
#include "core/renderer/targets/render_target.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/scene.hpp"
#include <type_traits>

namespace mufflon::renderer {

template < Device dev, class TL >
class RendererBase : public IRenderer {
public:
	static_assert(IsInstanceOf<TL, TargetList>::value,
				  "Renderer base class must be specialized with TargetList");

	using OutputHandlerType = typename TL::OutputHandlerType;
	using RenderBufferType = typename OutputHandlerType::template RenderBufferType<dev>;

	static constexpr Device DEVICE = dev;

	RendererBase(mufflon::scene::WorldContainer& world,
				 std::vector<::mufflon::scene::AttributeIdentifier> vertexAttribs = {},
				 std::vector<::mufflon::scene::AttributeIdentifier> faceAttribs = {},
				 std::vector<::mufflon::scene::AttributeIdentifier> sphereAttribs = {}) :
		IRenderer{ world },
		m_vertexAttribs(std::move(vertexAttribs)),
		m_faceAttribs(std::move(faceAttribs)),
		m_sphereAttribs(std::move(sphereAttribs))
	{
		if constexpr(dev == Device::CUDA)
			m_sceneDesc = make_udevptr<Device::CUDA, mufflon::scene::SceneDescriptor<Device::CUDA>>();
	}

	bool uses_device(Device device) const noexcept override { return may_use_device(device); }
	static constexpr bool may_use_device(Device device) noexcept { return DEVICE == device; }

	bool pre_iteration(IOutputHandler& outputBuffer) override {
		auto& out = dynamic_cast<OutputHandlerType&>(outputBuffer);

		const bool needsReset = this->get_reset_event() != ResetEvent::NONE;
		m_outputBuffer = out.template begin_iteration<dev>(needsReset);
		this->m_currentIteration = out.get_current_iteration();
		if(needsReset) {
			this->pre_reset();
			if(this->m_currentScene == nullptr)
				throw std::runtime_error("No scene is set!");
			if constexpr(dev == Device::CUDA) {
				auto desc = this->m_currentScene->template get_descriptor<dev>(m_vertexAttribs, m_faceAttribs, m_sphereAttribs);
				copy(m_sceneDesc.get(), &desc, sizeof(desc));
			} else {
				m_sceneDesc = this->m_currentScene->template get_descriptor<dev>(m_vertexAttribs, m_faceAttribs, m_sphereAttribs);
			}
			this->clear_reset();
			return true;
		}
		return false;
	}

	void post_iteration(IOutputHandler& outputBuffer) override {
		auto& out = dynamic_cast<OutputHandlerType&>(outputBuffer);
		out.template end_iteration<dev>();
	}

	std::unique_ptr<IOutputHandler> create_output_handler(int width, int height) override {
		return std::make_unique<OutputHandlerType>(width, height);
	}

protected:
	RenderBufferType m_outputBuffer;
	// Hold tables with the vertex attributes to get the required data on
	// scene-descriptor getters.
	std::vector<::mufflon::scene::AttributeIdentifier> m_vertexAttribs;
	std::vector<::mufflon::scene::AttributeIdentifier> m_faceAttribs;
	std::vector<::mufflon::scene::AttributeIdentifier> m_sphereAttribs;

	// CPU gets the descriptor directly, everyone else gets a unique_ptr
	std::conditional_t<DEVICE == Device::CPU || DEVICE == Device::OPENGL, scene::SceneDescriptor<DEVICE>,
		unique_device_ptr<DEVICE, scene::SceneDescriptor<DEVICE>>> m_sceneDesc;
};

} // namespace mufflon::renderer