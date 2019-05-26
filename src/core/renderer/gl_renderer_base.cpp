#include "gl_renderer_base.h"
#include "core/opengl/program_builder.h"
#include <glad/glad.h>

namespace mufflon::renderer {

GlRendererBase::GlRendererBase() {
    // compute shader that copies a texture into a buffer
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

void GlRendererBase::on_reset() {
	// color target
    glGenTextures(1, &m_colorTarget);
	glBindTexture(GL_TEXTURE_2D, m_colorTarget);
	glTextureStorage2D(m_colorTarget, 1, GL_RGBA32F, m_outputBuffer.get_width(), m_outputBuffer.get_height());

    // TODO depth target

    // framebuffer
	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0, m_colorTarget, 0);
	const auto fbStatus = glCheckNamedFramebufferStatus(m_framebuffer, GL_DRAW_FRAMEBUFFER);
	mAssert(fbStatus == GL_FRAMEBUFFER_COMPLETE);
	const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };
	glNamedFramebufferDrawBuffers(m_framebuffer, 1, attachments);
}

void GlRendererBase::begin_frame(ei::Vec4 clearColor) {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());
	glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT);
}

void GlRendererBase::end_frame() {
    // copy colorTarget to render target ssbo
	glUseProgram(m_copyShader);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_colorTarget);
	const auto dstBuf = m_outputBuffer.m_targets[RenderTargets::RADIANCE];
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dstBuf.id);
	glProgramUniform2ui(m_copyShader, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());

	glDispatchCompute(
        get_aligned(m_outputBuffer.get_width(), WORK_GROUP_SIZE),
        get_aligned(m_outputBuffer.get_height(), WORK_GROUP_SIZE),
		1
	);

	glFlush();
}

size_t GlRendererBase::get_aligned(size_t size, size_t alignment) {
	return size / alignment + !!(size % alignment);
}
}
