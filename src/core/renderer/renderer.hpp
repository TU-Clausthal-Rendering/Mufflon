#pragma once

#include "parameter.hpp"
#include "util/flag.hpp"
#include "util/string_view.hpp"
#include "util/type_helpers.hpp"
#include "core/renderer/targets/output_handler.hpp"
#include "core/scene/handles.hpp"
#include <ei/vector.hpp>

namespace mufflon {

enum class Device : unsigned char;

namespace renderer {

// Specifies what kind of reset happened
struct ResetEvent : public util::Flags<u16> {
	static constexpr u16 NONE				= 0b00'0000'0000;
	static constexpr u16 ANIMATION			= 0b00'0000'0001;
	static constexpr u16 CAMERA				= 0b00'0000'0010;
	static constexpr u16 LIGHT				= 0b00'0000'0100;
	static constexpr u16 TESSELLATION		= 0b00'0000'1000;
	static constexpr u16 SCENARIO			= 0b00'0001'0000;
	static constexpr u16 WORLD				= 0b00'0010'0000;
	static constexpr u16 PARAMETER			= 0b00'0100'0000;
	static constexpr u16 MANUAL				= 0b00'1000'0000;
	static constexpr u16 RENDERTARGET		= 0b01'0000'0000;
	static constexpr u16 RENDERER_ENABLE	= 0b10'0000'0000;

	static constexpr u16 ALL				= 0b11'1111'1111;

	// Maps certain possible state changes to reset events (excluding parameter changes)
	bool resolution_changed() const noexcept {
		return is_any_set( ResetEvent::SCENARIO
						 | ResetEvent::WORLD );
	}

	bool geometry_changed() const noexcept {
		return is_any_set( ResetEvent::SCENARIO
						 | ResetEvent::WORLD
						 | ResetEvent::ANIMATION
						 | ResetEvent::TESSELLATION );
	}

	bool lighttree_changed() const noexcept {
		return geometry_changed() || is_set(ResetEvent::LIGHT);
	}
};

inline std::string to_string(const ResetEvent evt) {
	std::string str;
	if(evt.is_set(ResetEvent::ANIMATION)) str += " ANIMATION";
	if(evt.is_set(ResetEvent::CAMERA)) str += " CAMERA";
	if(evt.is_set(ResetEvent::LIGHT)) str += " LIGHT";
	if(evt.is_set(ResetEvent::TESSELLATION)) str += " TESSELLATION";
	if(evt.is_set(ResetEvent::SCENARIO)) str += " SCENARIO";
	if(evt.is_set(ResetEvent::WORLD)) str += " WORLD";
	if(evt.is_set(ResetEvent::PARAMETER)) str += " PARAMETER";
	if(evt.is_set(ResetEvent::MANUAL)) str += " MANUAL";
	if(evt.is_set(ResetEvent::RENDERTARGET)) str += " RENDERTARGET";
	return str;
}

class IParameterHandler;

class IRenderer {
public:
	IRenderer() = default;
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
		m_lastReset.set(ResetEvent::ANIMATION);
	}

	// Triggers prior/after the scene's camera has been changed
	virtual void on_camera_changing() {}
	virtual void on_camera_changed() {
		m_lastReset.set(ResetEvent::CAMERA);
	}

	// Triggers when something about lights changes: position, intensity etc.
	virtual void on_light_changed() {
		m_lastReset.set(ResetEvent::LIGHT);
	}

	// Triggers when tessellation level has been requested
	virtual void on_tessellation_changing() {}
	virtual void on_tessellation_changed() {
		m_lastReset.set(ResetEvent::TESSELLATION);
	}

	/* Triggers prior/after the scenario (and thus the scene) has been changed.
	 * A separate scene loading event will be fired in-between.
	 */
	virtual void on_scenario_changing() {}
	virtual void on_scenario_changed(scene::ConstScenarioHandle newScenario) {
		m_lastReset.set(ResetEvent::SCENARIO);
	}

	// Triggers prior/after a new world has been loaded
	virtual void on_world_clearing() {
		m_lastReset.set(ResetEvent::WORLD);
	}

	// Triggers when a manual iteration reset has been requested
	virtual void on_manual_reset() {
		m_lastReset.set(ResetEvent::MANUAL);
	}

	virtual void on_render_target_changed() {
		m_lastReset.set(ResetEvent::RENDERTARGET);
	}

	// Triggers when a renderer parameter has changed
	virtual void on_renderer_parameter_changed() {
		m_lastReset.set(ResetEvent::PARAMETER);
	}

	// Triggered if this renderer is switched to from another renderer
	virtual void on_renderer_enable() {
		m_lastReset.set(ResetEvent::RENDERER_ENABLE);
	}

	// Gets called after the renderer reset is executed, ie. after
	// the new descriptor has been fetched
	void clear_reset() {
		this->post_reset();
		m_lastReset.clear_all();
	}

	void load_scene(scene::SceneHandle scene) {
		m_currentScene = scene;
		m_lastReset.set(ResetEvent::SCENARIO);
	}
	bool has_scene() const noexcept {
		return m_currentScene != nullptr;
	}

	ResetEvent get_reset_event() const noexcept { return m_lastReset; }

	// To avoid templatization of the base class, renderers have to
	// upcast the output handler to their expected input
	virtual bool pre_iteration(IOutputHandler& outputBuffer) = 0;
	virtual void post_iteration(IOutputHandler& outputBuffer) = 0;

	// TODO: should the renderer hold onto the output handler?
	virtual std::unique_ptr<IOutputHandler> create_output_handler(int width, int height) = 0;

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
	ResetEvent m_lastReset;
};

}} // namespace mufflon::renderer
