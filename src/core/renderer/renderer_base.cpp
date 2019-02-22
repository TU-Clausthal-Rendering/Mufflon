#include "renderer_base.hpp"
#include "core/scene/scene.hpp"
#include <stdexcept>

namespace mufflon::renderer {

template < Device dev >
RendererBase<dev>::RendererBase() {
	m_sceneDesc = make_udevptr<dev, mufflon::scene::SceneDescriptor<dev>>();
}
RendererBase<Device::CPU>::RendererBase() = default;

template < Device dev >
bool RendererBase<dev>::pre_iteration(OutputHandler& outputBuffer) {
	m_outputBuffer = outputBuffer.begin_iteration<dev>(m_reset);
	if(m_reset) {
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		auto desc = m_currentScene->get_descriptor<dev>({}, {}, {}, outputBuffer.get_resolution());
		copy(m_sceneDesc.get(), &desc, sizeof(desc));
		this->on_descriptor_requery();
		m_reset = false;
		return true;
	}
	return false;
}

bool RendererBase<Device::CPU>::pre_iteration(OutputHandler& outputBuffer) {
	m_outputBuffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	if(m_reset) {
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, outputBuffer.get_resolution());
		this->on_descriptor_requery();
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

} // namespace mufflon::renderer