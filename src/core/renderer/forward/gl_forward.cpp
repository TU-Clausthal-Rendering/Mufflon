#include "gl_forward.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"

namespace mufflon::renderer {

GlForward::GlForward() {

}

void GlForward::on_descriptor_requery() {

}

void GlForward::on_reset() {
	GlRendererBase::on_reset();
	/*m_copyShader = gl::ProgramBuilder().add_source(gl::ShaderType::Vertex, R"(
    #version 460
	void main() {
	    vec4 vertex = vec4(0.0, 0.0, 0.0, 1.0);
	    if(gl_VertexID == 0u) vertex = vec4(1.0, -1.0, 0.0, 1.0);
	    if(gl_VertexID == 1u) vertex = vec4(-1.0, -1.0, 0.0, 1.0);
	    if(gl_VertexID == 2u) vertex = vec4(1.0, 1.0, 0.0, 1.0);
	    if(gl_VertexID == 3u) vertex = vec4(-1.0, 1.0, 0.0, 1.0);
	    gl_Position = vertex;
    })").add_source(gl::ShaderType::Fragment, R"(
    #version 460    
    layout(location = 0) out vec4 out_fragColor;    

    void main() {
        out_fragColor = vec4(1.0f);
    }
    )").build();*/
	
}

void GlForward::iterate() {
	begin_frame({ 0.0f, 0.0f, 1.0f, 1.0f });

	end_frame();
}

} // namespace mufflon::renderer