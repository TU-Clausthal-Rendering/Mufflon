#pragma once

#include "parameter.hpp"
#include "core/scene/handles.hpp"
#include <ei/vector.hpp>
#include <string_view>

namespace mufflon {

enum class Device : unsigned char;

} // namespace mufflon

namespace mufflon { namespace renderer {

class OutputHandler; // TODO: implement an output handler for various configurations (variance, guides, ...)
class IParameterHandler;

class IRenderer {
public:
	virtual void iterate(OutputHandler& output) = 0;
	virtual void reset() = 0;
	virtual IParameterHandler& get_parameters() = 0;
	virtual bool has_scene() const noexcept = 0;
	virtual void load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) = 0;
	virtual std::string_view get_name() const noexcept = 0u;

	static bool uses_device(Device dev) noexcept { return false; }
};

}} // namespace mufflon::renderer