#pragma once

#include "parameter.hpp"
#include "core/scene/handles.hpp"
#include <ei/vector.hpp>
#include "util/string_view.hpp"

namespace mufflon {

enum class Device : unsigned char;

namespace renderer {

class OutputHandler;
class IParameterHandler;

class IRenderer {
public:
	virtual ~IRenderer() = default;

	virtual void iterate() = 0;
	virtual IParameterHandler& get_parameters() = 0;
	virtual StringView get_name() const noexcept = 0u;
	virtual StringView get_short_name() const noexcept = 0u;
	virtual bool uses_device(Device dev) noexcept = 0u;
	
	void reset() {
		m_reset = true;
	}

	void load_scene(mufflon::scene::SceneHandle scene, const ei::IVec2& resolution) {
		m_currentScene = scene;
		this->on_scene_load();
		this->reset();
	}

	bool has_scene() const noexcept {
		return m_currentScene != nullptr;
	}

	// Customizable operations for each renderer to react to events
	virtual void on_reset() {}
	virtual void on_descriptor_requery() {}
	virtual void on_scene_load() {}
	// Returns whether the scene was reset
	virtual bool pre_iteration(OutputHandler& outputBuffer) = 0;
	virtual void post_iteration(OutputHandler& outputBuffer) = 0;

protected:
	bool m_reset = true;
	mufflon::scene::SceneHandle m_currentScene = nullptr;
	int m_currentIteration = 0;
};

}} // namespace mufflon::renderer