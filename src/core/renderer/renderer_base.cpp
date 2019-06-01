#include "renderer_base.hpp"
#include "core/scene/scene.hpp"
#include <stdexcept>

namespace mufflon::renderer {

template < Device dev >
RendererBase<dev>::RendererBase() {
	m_sceneDesc = make_udevptr<dev, mufflon::scene::SceneDescriptor<dev>>();
}

template <>
RendererBase<Device::CPU>::RendererBase() {}

template <>
RendererBase<Device::OPENGL>::RendererBase() {}

template < Device dev >
bool RendererBase<dev>::pre_iteration(OutputHandler& outputBuffer) {
	m_outputBuffer = outputBuffer.begin_iteration<dev>(m_reset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if(m_reset) {
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		auto desc = m_currentScene->get_descriptor<dev>({}, {}, {});
		copy(m_sceneDesc.get(), &desc, sizeof(desc));
		this->on_descriptor_requery();
		this->on_reset();
		m_reset = false;
		return true;
	}
	return false;
}

template <>
bool RendererBase<Device::CPU>::pre_iteration(OutputHandler& outputBuffer) {
	m_outputBuffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if(m_reset) {
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {});
		this->on_descriptor_requery();
		this->on_reset();
		m_reset = false;
		return true;
	}
	return false;
}

template <>
bool RendererBase<Device::OPENGL>::pre_iteration(OutputHandler& outputBuffer) {
	m_outputBuffer = outputBuffer.begin_iteration<Device::OPENGL>(m_reset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if (m_reset) {
		if (m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		m_sceneDesc = m_currentScene->get_descriptor<Device::OPENGL>({}, {}, {});
		m_outputTargets = outputBuffer.get_target();
	    this->on_descriptor_requery();
		this->on_reset();
		m_reset = false;
		return true;
	}
	return false;
}

template < Device dev >
void RendererBase<dev>::post_iteration(OutputHandler& outputBuffer) {
	outputBuffer.end_iteration<dev>();
}

template class RendererBase<Device::CPU>;
template class RendererBase<Device::CUDA>;
template class RendererBase<Device::OPENGL>;

} // namespace mufflon::renderer