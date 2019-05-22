#pragma once
#include "core/scene/textures/format.hpp"
#include "gl_wrapper.hpp"

namespace mufflon {
namespace gl {
	struct TextureHandle
	{
		Handle id;
		u16 width;
		u16 height;
		u16 depth;
		scene::textures::Format format;
	};

}
}
