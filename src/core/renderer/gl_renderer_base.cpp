#include "gl_renderer_base.h"
#include "core/opengl/program_builder.h"
#include <glad/glad.h>
#include "core/opengl/gl_context.h"
#include "core/scene/scene.hpp"

namespace mufflon::renderer {

GlRendererBase::GlRendererBase(bool useDepth, bool useStencil) {
    // compute shader that copies a texture into a buffer
	m_copyShader = gl::ProgramBuilder()
        .add_file("shader/copy_output.glsl")
        .build_shader(gl::ShaderType::Compute)
        .build_program();

    // appropriate depth stencil format
    if(useDepth) {
		if(useStencil) {
			m_depthStencilFormat = GL_DEPTH24_STENCIL8;
			m_depthAttachmentType = GL_DEPTH_STENCIL_ATTACHMENT;
		}
		else {
			m_depthStencilFormat = GL_DEPTH_COMPONENT32F;
			m_depthAttachmentType = GL_DEPTH_ATTACHMENT;
		}
    } else {
		if(useStencil) {
			m_depthStencilFormat = GL_STENCIL_INDEX8;
			m_depthAttachmentType = GL_STENCIL_ATTACHMENT;
		}
		else {
			m_depthStencilFormat = 0;
			m_depthAttachmentType = 0;
		}
	}
}

void GlRendererBase::post_reset() {
	// Check if the resolution might have changed
	if(this->resolution_changed() || !m_framebuffer) {
		// create requested color targets
		uint32_t curTarget = 0;
		for(auto t : OutputValue::iterator) {
			//if(t & m_outputTargets) {
			glGenTextures(1, &m_colorTargets[curTarget]);
			glBindTexture(GL_TEXTURE_2D, m_colorTargets[curTarget]);
			glTextureStorage2D(m_colorTargets[curTarget], 1, GL_RGBA32F, m_outputBuffer.get_width(), m_outputBuffer.get_height());
			//}
			curTarget++;
		}

		// additional depth/stencil attachment
		if(m_depthStencilFormat) {
			glGenTextures(1, &m_depthTarget);
			glBindTexture(GL_TEXTURE_2D, m_depthTarget);
			glTextureStorage2D(m_depthTarget, 1, m_depthStencilFormat, m_outputBuffer.get_width(), m_outputBuffer.get_height());
		}

		// framebuffer
		glGenFramebuffers(1, &m_framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
		curTarget = 0;
		std::vector<GLenum> attachments;
		for(auto t : OutputValue::iterator) {
			//if(t & m_outputTargets) {
			glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0 + curTarget, m_colorTargets[curTarget], 0);
			attachments.push_back(GL_COLOR_ATTACHMENT0 + curTarget);
			//}
			curTarget++;
		}
		if(m_depthStencilFormat)
			glNamedFramebufferTexture(m_framebuffer, m_depthAttachmentType, m_depthTarget, 0);

		const auto fbStatus = glCheckNamedFramebufferStatus(m_framebuffer, GL_DRAW_FRAMEBUFFER);
		mAssert(fbStatus == GL_FRAMEBUFFER_COMPLETE);
		glNamedFramebufferDrawBuffers(m_framebuffer, GLsizei(attachments.size()), attachments.data());
	}
}

void GlRendererBase::begin_frame(ei::Vec4 clearColor) {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());
	glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void GlRendererBase::end_frame() {
	// copy render targets to buffers
    glUseProgram(m_copyShader);
	glProgramUniform2ui(m_copyShader, 0, m_outputBuffer.get_width(), m_outputBuffer.get_height());

	size_t curTarget = 0;
	for(auto t : OutputValue::iterator) {
		if(t & m_outputTargets) {
			glBindTextureUnit(0, m_colorTargets[curTarget]);
			const auto dstBuf = m_outputBuffer.m_targets[curTarget];
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dstBuf.id);

			glDispatchCompute(
				int(get_aligned(m_outputBuffer.get_width(), WORK_GROUP_SIZE)),
				int(get_aligned(m_outputBuffer.get_height(), WORK_GROUP_SIZE)),
				1
			);
		}
		curTarget++;
	}

	glFlush();
}

void GlRendererBase::draw_triangles(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);

	if(attribs & Attribute::Material) {
		mAssert(m_sceneDesc.materials.id);
		mAssert(!m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_sceneDesc.materials.id);
	}

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.polygon.numTriangles) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex and index buffer
        if(attribs & Attribute::Position) {
			mAssert(lod.polygon.vertices.id);
			glBindVertexBuffer(0, lod.polygon.vertices.id, lod.polygon.vertices.offset, sizeof(ei::Vec3));
        }
		if(attribs & Attribute::Normal) {
			mAssert(lod.polygon.normals.id);
			glBindVertexBuffer(1, lod.polygon.normals.id, lod.polygon.normals.offset, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Texcoord) {
			mAssert(lod.polygon.uvs.id);
			glBindVertexBuffer(2, lod.polygon.uvs.id, lod.polygon.uvs.offset, sizeof(ei::Vec2));
		}
		if(attribs & Attribute::Material) {
			mAssert(lod.polygon.matIndices.id);
			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, lod.polygon.matIndices.id,
				lod.polygon.matIndices.offset,
				lod.polygon.numTriangles * sizeof(u16));
		}

	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod.polygon.vertexIndices.id);
        
	    // draw
		glDrawElements(GLenum(pipe.topology), lod.polygon.numTriangles * 3, GL_UNSIGNED_INT, nullptr);
	}
}

void GlRendererBase::draw_spheres(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);

    if(attribs & Attribute::Material) {
		mAssert(m_sceneDesc.materials.id);
		mAssert(!m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_sceneDesc.materials.id);
    }

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.spheres.numSpheres) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex buffer
        if(attribs & Attribute::Position) {
			mAssert(lod.spheres.spheres.id);
			glBindVertexBuffer(0, lod.spheres.spheres.id, lod.spheres.spheres.offset, sizeof(ei::Sphere));
        }
        if(attribs & Attribute::Material) {
			mAssert(lod.spheres.matIndices.id);
			glBindVertexBuffer(1, lod.spheres.matIndices.id, lod.spheres.matIndices.offset, sizeof(u16));
        }

		// draw
		glDrawArrays(GLenum(pipe.topology), 0, lod.spheres.numSpheres);
	}
}

void GlRendererBase::draw_quads(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	mAssert(pipe.patch.vertices == 4);

	if(attribs & Attribute::Material) {
		mAssert(m_sceneDesc.materials.id);
		mAssert(!m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_sceneDesc.materials.id);
	}

	for(size_t i = 0; i < m_sceneDesc.numInstances; ++i) {
		const auto idx = m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = m_sceneDesc.lods[idx];

		if(!lod.polygon.numQuads) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&m_sceneDesc.instanceToWorld[i]));

		// bind vertex and index buffer
		if(attribs & Attribute::Position) {
			mAssert(lod.polygon.vertices.id);
			glBindVertexBuffer(0, lod.polygon.vertices.id, lod.polygon.vertices.offset, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Normal) {
			mAssert(lod.polygon.normals.id);
			glBindVertexBuffer(1, lod.polygon.normals.id, lod.polygon.normals.offset, sizeof(ei::Vec3));
		}
		if(attribs & Attribute::Texcoord) {
			mAssert(lod.polygon.uvs.id);
			glBindVertexBuffer(2, lod.polygon.uvs.id, lod.polygon.uvs.offset, sizeof(ei::Vec2));
		}
        if(attribs & Attribute::Material) {
			mAssert(lod.polygon.matIndices.id);
			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, lod.polygon.matIndices.id,
				lod.polygon.matIndices.offset,
				(lod.polygon.numTriangles + lod.polygon.numQuads) * sizeof(u16));
			glProgramUniform1ui(pipe.program, 2, lod.polygon.numTriangles);
        }

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod.polygon.vertexIndices.id);
        
		// draw
		size_t offset = lod.polygon.numTriangles * 3 * sizeof(GLuint);
		glDrawElements(GLenum(pipe.topology), lod.polygon.numQuads * 4, GL_UNSIGNED_INT, reinterpret_cast<void*>(offset));
	}
}

GlRendererBase::CameraTransforms GlRendererBase::get_camera_transforms() const {
	CameraTransforms t;

	auto* cam = m_currentScene->get_camera();

	t.position = cam->get_position(0);
	t.direction = cam->get_view_dir(0);

    // windows stuff...
#undef near
#undef far
	t.near = cam->get_near();
	t.far = cam->get_far();
	t.screen.x = m_outputBuffer.get_width();
	t.screen.y = m_outputBuffer.get_height();
    
    float fov = 1.5f;
	if(auto pcam = dynamic_cast<const cameras::Pinhole*>(cam)) {
		fov = pcam->get_vertical_fov();
	}
	t.projection = ei::perspectiveGL(fov,
		float(t.screen.x) / t.screen.y,
		t.near, t.far);
	t.view = ei::camera(
		t.position,
		t.position + t.direction,
		cam->get_up_dir(0)
	);
	t.invView = ei::invert(t.view);
	t.viewProj = t.projection * t.view;
	// transpose since opengl expects column major
	t.projection = ei::transpose(t.projection);
	t.view = ei::transpose(t.view);
	t.viewProj = ei::transpose(t.viewProj);
	t.invView = ei::transpose(t.invView);

	return t;
}

size_t GlRendererBase::get_aligned(size_t size, size_t alignment) {
	return size / alignment + !!(size % alignment);
}
}
