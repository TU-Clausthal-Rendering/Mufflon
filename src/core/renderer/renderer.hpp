#pragma once

#include "parameter.hpp"
#include "core/scene/handles.hpp"
#include <ei/vector.hpp>
#include "util/string_view.hpp"

namespace mufflon {

enum class Device : unsigned char;

namespace renderer {

class OutputHandler; // TODO: implement an output handler for various configurations (variance, guides, ...)
class IParameterHandler;

class IRenderer {
public:
	virtual void iterate(OutputHandler& output) = 0;
	virtual void reset() = 0;
	virtual IParameterHandler& get_parameters() = 0;
	virtual bool has_scene() const noexcept = 0;
	virtual void load_scene(::mufflon::scene::SceneHandle scene, const ei::IVec2& resolution) = 0;
	virtual StringView get_name() const noexcept = 0u;
	virtual bool uses_device(Device dev) noexcept { return false; }
};

}} // namespace mufflon::renderer