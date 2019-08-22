#include "renderer_base.hpp"
#include "core/scene/scene.hpp"
#include <stdexcept>

namespace mufflon::renderer {

template < Device dev >
RendererBase<dev>::RendererBase(std::vector<const char*> vertexAttribs,
								std::vector<const char*> faceAttribs,
								std::vector<const char*> sphereAttribs) :
	m_vertexAttribs(std::move(vertexAttribs)),
	m_faceAttribs(std::move(faceAttribs)),
	m_sphereAttribs(std::move(sphereAttribs))
{}

template <>
RendererBase<Device::CUDA>::RendererBase(std::vector<const char*> vertexAttribs,
								std::vector<const char*> faceAttribs,
								std::vector<const char*> sphereAttribs) :
	m_vertexAttribs(std::move(vertexAttribs)),
	m_faceAttribs(std::move(faceAttribs)),
	m_sphereAttribs(std::move(sphereAttribs))
{
	m_sceneDesc = make_udevptr<Device::CUDA, mufflon::scene::SceneDescriptor<Device::CUDA>>();
}



template < Device dev >
bool RendererBase<dev>::pre_iteration(OutputHandler& outputBuffer) {
	const bool needsReset = get_reset_event() != ResetEvent::NONE;
	m_outputBuffer = outputBuffer.begin_iteration<dev>(needsReset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if(needsReset) {
		this->pre_reset();
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		auto desc = m_currentScene->get_descriptor<dev>(m_vertexAttribs, m_faceAttribs, m_sphereAttribs);
		copy(m_sceneDesc.get(), &desc, sizeof(desc));
		this->clear_reset();
		return true;
	}
	return false;
}

template <>
bool RendererBase<Device::CPU>::pre_iteration(OutputHandler& outputBuffer) {
	const bool needsReset = get_reset_event() != ResetEvent::NONE;
	m_outputBuffer = outputBuffer.begin_iteration<Device::CPU>(needsReset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if(needsReset) {
		this->pre_reset();
		if(m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>(m_vertexAttribs, m_faceAttribs, m_sphereAttribs);
		this->clear_reset();
		return true;
	}
	return false;
}

template <>
bool RendererBase<Device::OPENGL>::pre_iteration(OutputHandler& outputBuffer) {
	const bool needsReset = get_reset_event() != ResetEvent::NONE;
	m_outputBuffer = outputBuffer.begin_iteration<Device::OPENGL>(needsReset);
	m_currentIteration = outputBuffer.get_current_iteration();
	if (needsReset) {
		this->pre_reset();
		if (m_currentScene == nullptr)
			throw std::runtime_error("No scene is set!");
		m_sceneDesc = m_currentScene->get_descriptor<Device::OPENGL>(m_vertexAttribs, m_faceAttribs, m_sphereAttribs);
		m_outputTargets = outputBuffer.get_target();
		this->clear_reset();
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