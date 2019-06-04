#pragma once
#include "core/scene/textures/format.hpp"
#include "gl_wrapper.hpp"

namespace mufflon {
namespace gl {
	struct TextureHandle
	{
		Handle id = 0;
		u16 width = 0;
		u16 height = 0;
		u16 depth = 0;
		scene::textures::Format format;
	};

}
}
