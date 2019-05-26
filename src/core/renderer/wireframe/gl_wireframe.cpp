#include "gl_wireframe.h"
#include <glad/glad.h>

namespace mufflon::renderer {

void GlWireframe::on_reset() {
	GlRendererBase::on_reset();
}

void GlWireframe::iterate() {
	begin_frame({ 1.0f, 0.0f, 0.0f, 1.0f });



	end_frame();
}
}
