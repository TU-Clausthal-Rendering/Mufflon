#include "gl_renderer_base.h"
#include "core/opengl/program_builder.h"
#include <glad/glad.h>
#include "core/opengl/gl_context.h"
#include "core/scene/scene.hpp"

namespace mufflon::renderer {

GlRendererBase::GlRendererBase() {
    // compute shader that copies a texture into a buffer
	m_copyShader = gl::ProgramBuilder()
        .add_file("shader/copy_output.glsl")
        .build_shader(gl::ShaderType::Compute)
        .build_program();
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
        int(get_aligned(m_outputBuffer.get_width(), WORK_GROUP_SIZE)),
        int(get_aligned(m_outputBuffer.get_height(), WORK_GROUP_SIZE)),
		1
	);

	glFlush();
}

void GlRendererBase::draw_triangles(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.polygon.numTriangles) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex and index buffer
        if(attribs & Attribute::Position) {
			mAssert(lod.polygon.vertices.id);
			glBindVertexBuffer(0, lod.polygon.vertices.id, 0, sizeof(ei::Vec3));
        }
		if(attribs & Attribute::Normal) {
			mAssert(lod.polygon.normals.id);
			glBindVertexBuffer(1, lod.polygon.normals.id, 0, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Texcoord) {
			mAssert(lod.polygon.uvs.id);
			glBindVertexBuffer(2, lod.polygon.uvs.id, 0, sizeof(ei::Vec2));
		}

	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod.polygon.vertexIndices.id);
        
	    // draw
		glDrawElements(GL_TRIANGLES, lod.polygon.numTriangles * 3, GL_UNSIGNED_INT, nullptr);
	}
}

void GlRendererBase::draw_spheres(const gl::Pipeline& pipe) {
	gl::Context::set(pipe);

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.spheres.numSpheres) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex and index buffer
		mAssert(lod.spheres.spheres.id);
		glBindVertexBuffer(0, lod.spheres.spheres.id, 0, sizeof(ei::Sphere));

		// draw
		glDrawArrays(GL_POINTS, 0, lod.spheres.numSpheres);
	}
}

void GlRendererBase::draw_quads(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	mAssert(pipe.patch.vertices == 4);

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.polygon.numQuads) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex and index buffer
		if(attribs & Attribute::Position) {
			mAssert(lod.polygon.vertices.id);
			glBindVertexBuffer(0, lod.polygon.vertices.id, 0, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Normal) {
			mAssert(lod.polygon.normals.id);
			glBindVertexBuffer(1, lod.polygon.normals.id, 0, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Texcoord) {
			mAssert(lod.polygon.uvs.id);
			glBindVertexBuffer(2, lod.polygon.uvs.id, 0, sizeof(ei::Vec2));
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod.polygon.vertexIndices.id);
        
		// draw
		size_t offset = lod.polygon.numTriangles * 3 * sizeof(GLuint);
		glDrawElements(GL_PATCHES, lod.polygon.numQuads * 4, GL_UNSIGNED_INT, reinterpret_cast<void*>(offset));
	}
}

GlRendererBase::CameraTransforms GlRendererBase::get_camera_transforms() const {
	CameraTransforms t;

	auto* cam = m_currentScene->get_camera();
	float fov = 1.5f;
	if(auto pcam = dynamic_cast<const cameras::Pinhole*>(cam)) {
		fov = pcam->get_vertical_fov();
	}
	t.projection = ei::perspectiveGL(fov,
		float(m_outputBuffer.get_width()) / m_outputBuffer.get_height(),
		cam->get_near(), cam->get_far());
	t.view = ei::camera(
		cam->get_position(0),
		cam->get_position(0) + cam->get_view_dir(0),
		cam->get_up_dir(0)
	);
	t.viewProj = t.projection * t.view;
	// transpose since opengl expects column major
	t.projection = ei::transpose(t.projection);
	t.view = ei::transpose(t.view);
	t.viewProj = ei::transpose(t.viewProj);

	return t;
}

size_t GlRendererBase::get_aligned(size_t size, size_t alignment) {
	return size / alignment + !!(size % alignment);
}
}
