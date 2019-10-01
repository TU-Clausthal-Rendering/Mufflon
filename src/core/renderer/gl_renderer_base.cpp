#include "gl_renderer_base.hpp"
#include <glad/glad.h>
// windows stuff...
#undef near
#undef far
#include "core/opengl/program_builder.hpp"
#include "core/opengl/gl_context.hpp"
#include "core/scene/scene.hpp"
#include "core/renderer/forward/forward_params.hpp"
#include "core/renderer/wireframe/wireframe_params.hpp"
#include <ei/vector.hpp>


namespace mufflon::renderer {

GlRenderer::GlRenderer(const u32 colorTargetCount, bool useDepth, bool useStencil) :
	m_colorTargets(colorTargetCount)
{
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
		} else {
			m_depthStencilFormat = GL_DEPTH_COMPONENT32F;
			m_depthAttachmentType = GL_DEPTH_ATTACHMENT;
		}
	} else {
		if(useStencil) {
			m_depthStencilFormat = GL_STENCIL_INDEX8;
			m_depthAttachmentType = GL_STENCIL_ATTACHMENT;
		} else {
			m_depthStencilFormat = 0;
			m_depthAttachmentType = 0;
		}
	}
}

void GlRenderer::reset(int width, int height) {
	// create requested color targets
	for(auto& target : m_colorTargets) {
		// TODO: Check if the target is enabled
		glGenTextures(1, &target);
		glBindTexture(GL_TEXTURE_2D, target);
		glTextureStorage2D(target, 1, GL_RGBA32F, width, height);
	}

	// additional depth/stencil attachment
	if(m_depthStencilFormat) {
		glGenTextures(1, &m_depthTarget);
		glBindTexture(GL_TEXTURE_2D, m_depthTarget);
		glTextureStorage2D(m_depthTarget, 1, m_depthStencilFormat, width, height);
	}

	// framebuffer
	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	std::vector<GLenum> attachments;
	for(u32 i = 0u; i < static_cast<u32>(m_colorTargets.size()); ++i) {
		glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0 + i, m_colorTargets[i], 0);
		attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
	}
	if(m_depthStencilFormat)
		glNamedFramebufferTexture(m_framebuffer, m_depthAttachmentType, m_depthTarget, 0);

	const auto fbStatus = glCheckNamedFramebufferStatus(m_framebuffer, GL_DRAW_FRAMEBUFFER);
	mAssert(fbStatus == GL_FRAMEBUFFER_COMPLETE);
	glNamedFramebufferDrawBuffers(m_framebuffer, GLsizei(attachments.size()), attachments.data());
}

void GlRenderer::begin_frame(ei::Vec4 clearColor, int width, int height) {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_framebuffer);
	glViewport(0, 0, width, height);
	glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void GlRenderer::end_frame(int width, int height) {
	// copy render targets to buffers
    glUseProgram(m_copyShader);
	glProgramUniform2ui(m_copyShader, 0, width, height);

	for(u32 i = 0u; i < static_cast<u32>(m_colorTargets.size()); ++i) {
		auto target = this->get_target(i);
		if(target.has_value()) {
			glBindTextureUnit(0, m_colorTargets[i]);
			const auto dstBuf = target.value();
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dstBuf.id);

			glDispatchCompute(
				int(get_aligned(width, WORK_GROUP_SIZE)),
				int(get_aligned(height, WORK_GROUP_SIZE)),
				1
			);
		}
	}

	glFlush();
}

void GlRenderer::draw_triangles(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	const auto& sceneDesc = this->get_scene_descriptor();

	if(attribs & Attribute::Material) {
		mAssert(sceneDesc.materials.id);
		mAssert(!sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sceneDesc.materials.id);
	}

	for(size_t i = 0; i < sceneDesc.numInstances; ++i) {
		const auto idx = sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = sceneDesc.lods[idx];

		if(!lod.polygon.numTriangles) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&sceneDesc.instanceToWorld[i]));

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

void GlRenderer::draw_spheres(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	const auto& sceneDesc = this->get_scene_descriptor();

    if(attribs & Attribute::Material) {
		mAssert(sceneDesc.materials.id);
		mAssert(!sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sceneDesc.materials.id);
    }

	for(size_t i = 0; i < sceneDesc.numInstances; ++i) {
		const auto idx = sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = sceneDesc.lods[idx];

		if(!lod.spheres.numSpheres) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&sceneDesc.instanceToWorld[i]));

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

void GlRenderer::draw_quads(const gl::Pipeline& pipe, Attribute attribs) {
	gl::Context::set(pipe);
	const auto& sceneDesc = this->get_scene_descriptor();
	mAssert(pipe.patch.vertices == 4);

	if(attribs & Attribute::Material) {
		mAssert(sceneDesc.materials.id);
		mAssert(!sceneDesc.materials.offset);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sceneDesc.materials.id);
	}

	for(size_t i = 0; i < sceneDesc.numInstances; ++i) {
		const auto idx = sceneDesc.lodIndices[i];
		const scene::LodDescriptor<Device::OPENGL>& lod = sceneDesc.lods[idx];

		if(!lod.polygon.numQuads) continue;

		// Set the instance transformation matrix
		glProgramUniformMatrix4x3fv(pipe.program, 1, 1, GL_TRUE, reinterpret_cast<const float*>(&sceneDesc.instanceToWorld[i]));

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

size_t GlRenderer::get_aligned(size_t size, size_t alignment) {
	return size / alignment + !!(size % alignment);
}

}
