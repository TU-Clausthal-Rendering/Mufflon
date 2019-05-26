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

	m_copyShader = gl::ProgramBuilder().add_source(gl::ShaderType::Vertex, R"(
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
    )").build();
}

void GlForward::unload() {
	gl::deleteTexture(m_colorTarget);
	m_colorTarget = 0;
	gl::deleteTexture(m_depthTarget);
	m_depthTarget = 0;
	glDeleteFramebuffers(1, &m_framebuffer);
	m_framebuffer = 0;
	glDeleteProgram(m_copyShader);
	m_copyShader = 0;
}

void GlForward::iterate() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);


	glFlush();
    // gl buffer handle with floats
	auto dstBuf = m_outputBuffer.m_targets[RenderTargets::RADIANCE];
	//glGetTexImage(m_colorTarget, 0, GL_RGB, GL_FLOAT, dstBuf);
	float val[3] = { 1.0f, 0.0f, 0.0f };
	gl::clearBufferData(dstBuf.id, m_outputBuffer.get_width() * m_outputBuffer.get_height(), &val);
}

} // namespace mufflon::renderer