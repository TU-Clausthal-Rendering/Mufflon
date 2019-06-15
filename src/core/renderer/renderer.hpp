#pragma once

#include "parameter.hpp"
#include "core/scene/handles.hpp"
#include <ei/vector.hpp>
#include "util/string_view.hpp"

namespace mufflon {

enum class Device : unsigned char;

namespace renderer {

// Specifies what kind of reset happened
enum class ResetEvent {
	NONE			= 0b000000000,
	ANIMATION		= 0b000000001,
	CAMERA			= 0b000000010,
	LIGHT			= 0b000000100,
	TESSELLATION	= 0b000001000,
	SCENARIO		= 0b000010000,
	WORLD			= 0b000100000,
	PARAMETER		= 0b001000000,
	MANUAL			= 0b010000000,
	RENDERTARGET	= 0b100000000
};

inline ResetEvent operator|(const ResetEvent a, const ResetEvent b) {
	return static_cast<ResetEvent>(static_cast<std::underlying_type_t<ResetEvent>>(a) |
								   static_cast<std::underlying_type_t<ResetEvent>>(b));
}

inline ResetEvent operator&(const ResetEvent a, const ResetEvent b) {
	return static_cast<ResetEvent>(static_cast<std::underlying_type_t<ResetEvent>>(a) &
								   static_cast<std::underlying_type_t<ResetEvent>>(b));
}

inline std::string to_string(const ResetEvent evt) {
	std::string str;
	if((evt & ResetEvent::ANIMATION) != ResetEvent::NONE) str += " ANIMATION";
	if((evt & ResetEvent::CAMERA) != ResetEvent::NONE) str += " CAMERA";
	if((evt & ResetEvent::LIGHT) != ResetEvent::NONE) str += " LIGHT";
	if((evt & ResetEvent::TESSELLATION) != ResetEvent::NONE) str += " TESSELLATION";
	if((evt & ResetEvent::SCENARIO) != ResetEvent::NONE) str += " SCENARIO";
	if((evt & ResetEvent::WORLD) != ResetEvent::NONE) str += " WORLD";
	if((evt & ResetEvent::PARAMETER) != ResetEvent::NONE) str += " PARAMETER";
	if((evt & ResetEvent::MANUAL) != ResetEvent::NONE) str += " MANUAL";
	if((evt & ResetEvent::RENDERTARGET) != ResetEvent::NONE) str += " RENDERTARGET";
	return str;
}

class OutputHandler;
class IParameterHandler;

class IRenderer {
public:
	virtual ~IRenderer() = default;

	virtual void iterate() = 0;
	virtual IParameterHandler& get_parameters() = 0;
	virtual StringView get_name() const noexcept = 0;
	virtual StringView get_short_name() const noexcept = 0;
	virtual bool uses_device(Device dev) const noexcept = 0;

	/* All reset/change events below SHOULD call the base function if
	 * overwritten unless you are well aware of what you're doing.
	 */

	/* Triggers prior/after a different animation frame has been set.
	 * A separate scene loading event will be fired in-between.
	 */
	virtual void on_animation_frame_changing(const u32 from, const u32 to) {}
	virtual void on_animation_frame_changed(const u32 from, const u32 to) {
		m_lastReset = m_lastReset | ResetEvent::ANIMATION;
	}

	// Triggers prior/after the scene's camera has been changed
	virtual void on_camera_changing() {}
	virtual void on_camera_changed() {
		m_lastReset = m_lastReset | ResetEvent::CAMERA;
	}

	// Triggers when something about lights changes: position, intensity etc.
	virtual void on_light_changed() {
		m_lastReset = m_lastReset | ResetEvent::LIGHT;
	}

	// Triggers when tessellation level has been requested
	virtual void on_tessellation_changing() {}
	virtual void on_tessellation_changed() {
		m_lastReset = m_lastReset | ResetEvent::TESSELLATION;
	}

	/* Triggers prior/after the scenario (and thus the scene) has been changed.
	 * A separate scene loading event will be fired in-between.
	 */
	virtual void on_scenario_changing() {}
	virtual void on_scenario_changed(scene::ConstScenarioHandle newScenario) {
		m_lastReset = m_lastReset | ResetEvent::SCENARIO;
	}

	// Triggers prior/after a new world has been loaded
	virtual void on_world_clearing() {
		m_lastReset = m_lastReset | ResetEvent::WORLD;
	}

	// Triggers when a manual iteration reset has been requested
	virtual void on_manual_reset() {
		m_lastReset = m_lastReset | ResetEvent::MANUAL;
	}

	virtual void on_render_target_changed() {
		m_lastReset = m_lastReset | ResetEvent::RENDERTARGET;
	}

	// Triggers when a renderer parameter has changed
	virtual void on_renderer_parameter_changed() {
		m_lastReset = m_lastReset | ResetEvent::PARAMETER;
	}

	// Gets called after the renderer reset is executed, ie. after
	// the new descriptor has been fetched
	void clear_reset() {
		this->post_reset();
		m_lastReset = {};
	}

	void load_scene(scene::SceneHandle scene) {
		m_currentScene = scene;
		m_lastReset = m_lastReset | ResetEvent::SCENARIO;
	}
	bool has_scene() const noexcept {
		return m_currentScene != nullptr;
	}

	ResetEvent get_reset_event() const noexcept { return m_lastReset; }

	// Maps certain possible state changes to reset events (excluding parameter changes)
	bool resolution_changed() const noexcept {
		return (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::SCENARIO))
			|| (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::WORLD));
	}

	bool geometry_changed() const noexcept {
		return (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::SCENARIO))
			|| (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::WORLD))
			|| (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::ANIMATION))
			|| (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::TESSELLATION));
	}

	bool lighttree_changed() const noexcept {
		return geometry_changed() || (static_cast<int>(get_reset_event()) & static_cast<int>(ResetEvent::LIGHT));
	}

	// Returns whether the scene was reset
	virtual bool pre_iteration(OutputHandler& outputBuffer) = 0;
	virtual void post_iteration(OutputHandler& outputBuffer) = 0;

protected:
	// Gets called before/after the renderer reset is executed, ie. before
	// the descriptor refetch, but after the output buffer reset
	virtual void pre_reset() {}

	// Gets called after the renderer reset is executed, ie. after
	// the new descriptor has been fetched
	virtual void post_reset() {}

	mufflon::scene::SceneHandle m_currentScene = nullptr;
	int m_currentIteration = 0;

private:
	ResetEvent m_lastReset = ResetEvent::NONE;
};

}} // namespace mufflon::renderer