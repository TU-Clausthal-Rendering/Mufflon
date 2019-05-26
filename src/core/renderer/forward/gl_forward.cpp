#include "gl_forward.hpp"
#include <glad/glad.h>
#include "core/opengl/program_builder.h"

namespace mufflon::renderer {

void GlForward::on_descriptor_requery() {

}

void GlForward::on_reset() {
	unload();
    // create textures
	m_colorTarget = gl::genTexture();
	glBindTexture(GL_TEXTURE_2D, m_colorTarget);
	glTextureStorage2D(m_colorTarget, 1, GL_RGBA32F, m_outputBuffer.get_width(), m_outputBuffer.get_height());

    // create framebuffer
	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0, m_colorTarget, 0);
	const auto fbStatus = glCheckNamedFramebufferStatus(m_framebuffer, GL_DRAW_FRAMEBUFFER);
	mAssert(fbStatus == GL_FRAMEBUFFER_COMPLETE);
	const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };
	glNamedFramebufferDrawBuffers(m_framebuffer, 1, attachments);

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

void GlForward::unload() {
	gl::deleteTexture(m_colorTarget);
	m_colorTarget = 0;
	gl::deleteTexture(m_depthTarget);
	m_depthTarget = 0;
	glDeleteFramebuffers(1, &m_framebuffer);
	m_framebuffer = 0;
}

GlForward::GlForward() {
	m_copyShader = gl::ProgramBuilder().add_source(gl::ShaderType::Compute, R"(   
    #version 460
    layout(local_size_x = 16, local_size_y  = 16) in;
    layout(binding = 0) uniform sampler2D src_image;
    layout(binding = 0) writeonly restrict buffer dst_buffer {
        float data[];
    };
    layout(location = 0) uniform uvec2 size;

    void main(){
        vec3 color = texelFetch(src_image, ivec2(gl_GlobalInvocationID), 0).rgb;
        if(gl_GlobalInvocationID.x >= size.x) return;
        if(gl_GlobalInvocationID.y >= size.y) return;

        uint index = gl_GlobalInvocationID.y * size.x + gl_GlobalInvocationID.x;
        data[3 * index] = color.r;
        data[3 * index + 1] = color.g;
        data[3 * index + 2] = color.b;
    }
    )").build();
}

void GlForward::iterate() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

    // copy image from texture to buffer
	glUseProgram(m_copyShader);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_colorTarget);
	auto dstBuf = m_outputBuffer.m_targets[RenderTargets::RADIANCE];
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dstBuf.id);
	glProgramUniform2ui(m_copyShader, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());

	glDispatchCompute(m_outputBuffer.get_width() / 16 + 1, m_outputBuffer.get_height() / 16 + 1, 1);

	glFlush();
}

} // namespace mufflon::renderer