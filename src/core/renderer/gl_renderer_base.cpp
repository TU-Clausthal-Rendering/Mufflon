#include "gl_renderer_base.h"
#include <glad/glad.h>
// windows stuff...
#undef near
#undef far
#include "core/opengl/program_builder.h"
#include "core/cameras/pinhole.hpp"
#include "core/opengl/gl_context.h"
#include "core/scene/scene.hpp"
#include "core/renderer/forward/forward_params.hpp"
#include "core/renderer/wireframe/wireframe_params.hpp"
#include <ei/vector.hpp>


namespace mufflon::renderer {

template < class TL >
GlRendererBase<TL>::GlRendererBase(bool useDepth, bool useStencil) {
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

template < class TL >
void GlRendererBase<TL>::post_reset() {
	// TODO
	// Check if the resolution might have changed
	if(this->get_reset_event().resolution_changed() || !m_framebuffer) {
		// create requested color targets
		for(u32 i = 0u; i < COLOR_ATTACHMENTS; ++i) {
			// TODO: Check if the target is enabled
			glGenTextures(1, &m_colorTargets[i]);
			glBindTexture(GL_TEXTURE_2D, m_colorTargets[i]);
			glTextureStorage2D(m_colorTargets[i], 1, GL_RGBA32F, this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
		}

		// additional depth/stencil attachment
		if(m_depthStencilFormat) {
			glGenTextures(1, &m_depthTarget);
			glBindTexture(GL_TEXTURE_2D, m_depthTarget);
			glTextureStorage2D(m_depthTarget, 1, m_depthStencilFormat, this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
		}

		// framebuffer
		glGenFramebuffers(1, &m_framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
		std::vector<GLenum> attachments;
		for(u32 i = 0u; i < COLOR_ATTACHMENTS; ++i) {
			glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0 + i, m_colorTargets[i], 0);
			attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
		}
		if(m_depthStencilFormat)
			glNamedFramebufferTexture(m_framebuffer, m_depthAttachmentType, m_depthTarget, 0);

		const auto fbStatus = glCheckNamedFramebufferStatus(m_framebuffer, GL_DRAW_FRAMEBUFFER);
		mAssert(fbStatus == GL_FRAMEBUFFER_COMPLETE);
		glNamedFramebufferDrawBuffers(m_framebuffer, GLsizei(attachments.size()), attachments.data());
	}
}

template < class TL >
void GlRendererBase<TL>::begin_frame(ei::Vec4 clearColor) {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());
	glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

template < class TL >
void GlRendererBase<TL>::end_frame() {
	// copy render targets to buffers
    glUseProgram(m_copyShader);
	glProgramUniform2ui(m_copyShader, 0, this->m_outputBuffer.get_width(), this->m_outputBuffer.get_height());

	for(u32 i = 0u; i < COLOR_ATTACHMENTS; ++i) {
		if(this->m_outputBuffer.is_target_enabled(i)) {
			glBindTextureUnit(0, m_colorTargets[i]);
			const auto dstBuf = this->m_outputBuffer.get_target(i);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dstBuf.id);

			glDispatchCompute(
				int(get_aligned(this->m_outputBuffer.get_width(), WORK_GROUP_SIZE)),
				int(get_aligned(this->m_outputBuffer.get_height(), WORK_GROUP_SIZE)),
				1
			);
		}
	}

	glFlush();
}

template < class TL >
void GlRendererBase<TL>::draw_triangles(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);

	if(attribs & Attribute::Material) {
		mAssert(this->m_sceneDesc.materials.id);
		mAssert(!this->m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->m_sceneDesc.materials.id);
	}

	for(size_t i = 0; i < this->m_sceneDesc.numInstances; ++i) {
		const auto idx = this->m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = this->m_sceneDesc.lods[idx];

		if(!lod.polygon.numTriangles) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&this->m_sceneDesc.instanceToWorld[i]));

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

template < class TL >
void GlRendererBase<TL>::draw_spheres(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);

    if(attribs & Attribute::Material) {
		mAssert(this->m_sceneDesc.materials.id);
		mAssert(!this->m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->m_sceneDesc.materials.id);
    }

	for(size_t i = 0; i < this->m_sceneDesc.numInstances; ++i) {
		const auto idx = this->m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = this->m_sceneDesc.lods[idx];

		if(!lod.spheres.numSpheres) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&this->m_sceneDesc.instanceToWorld[i]));

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

template < class TL >
void GlRendererBase<TL>::draw_quads(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	mAssert(pipe.patch.vertices == 4);

	if(attribs & Attribute::Material) {
		mAssert(this->m_sceneDesc.materials.id);
		mAssert(!this->m_sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->m_sceneDesc.materials.id);
	}

	for(size_t i = 0; i < this->m_sceneDesc.numInstances; ++i) {
		const auto idx = this->m_sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = this->m_sceneDesc.lods[idx];

		if(!lod.polygon.numQuads) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&this->m_sceneDesc.instanceToWorld[i]));

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

template < class TL >
typename GlRendererBase<TL>::CameraTransforms GlRendererBase<TL>::get_camera_transforms() const {
	CameraTransforms t;

	auto* cam = this->m_currentScene->get_camera();

	t.position = cam->get_position(0);
	t.direction = cam->get_view_dir(0);

	t.near = cam->get_near();
	t.far = cam->get_far();
	t.screen.x = this->m_outputBuffer.get_width();
	t.screen.y = this->m_outputBuffer.get_height();
    
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

template < class TL >
size_t GlRendererBase<TL>::get_aligned(size_t size, size_t alignment) {
	return size / alignment + !!(size % alignment);
}

// TODO: Find a solution for this!
template class GlRendererBase<ForwardTargets>;
template class GlRendererBase<WireframeTargets>;

}
