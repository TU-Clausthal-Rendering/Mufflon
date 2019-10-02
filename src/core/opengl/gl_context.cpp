#include "gl_context.hpp"
#include "util/assert.hpp"
#include <glad/glad.h>

namespace mufflon::gl {

void Context::set(const Pipeline& pipeline) {
	auto& state = get().m_state;

	mAssert(pipeline.framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pipeline.framebuffer);

    // Patches

    if(state.patch.vertices != pipeline.patch.vertices) {
		glPatchParameteri(GL_PATCH_VERTICES, pipeline.patch.vertices);
    }
    if(state.patch.inner != pipeline.patch.inner) {
		glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, state.patch.inner.data());
    }
	if(state.patch.outer != pipeline.patch.outer) {
		glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, state.patch.outer.data());
	}
	state.patch = pipeline.patch;

    // rasterizer

	if(state.rasterizer.cullMode != pipeline.rasterizer.cullMode)
	{
		if(pipeline.rasterizer.cullMode == CullMode::None)
			glDisable(GL_CULL_FACE);
		else {
			glEnable(GL_CULL_FACE);
			glCullFace(static_cast<GLenum>(pipeline.rasterizer.cullMode));
		}
		state.rasterizer.cullMode = pipeline.rasterizer.cullMode;
	}
	if(state.rasterizer.frontFaceWinding != pipeline.rasterizer.frontFaceWinding)
	{
		glFrontFace(static_cast<GLenum>(pipeline.rasterizer.frontFaceWinding));
		state.rasterizer.frontFaceWinding = pipeline.rasterizer.frontFaceWinding;
	}
	if(state.rasterizer.fillMode != pipeline.rasterizer.fillMode)
	{
		glPolygonMode(GL_FRONT_AND_BACK, static_cast<GLenum>(pipeline.rasterizer.fillMode));
		state.rasterizer.fillMode = pipeline.rasterizer.fillMode;
	}
	if(state.rasterizer.lineWidth != pipeline.rasterizer.lineWidth)
	{
		glLineWidth(pipeline.rasterizer.lineWidth);
		state.rasterizer.lineWidth = pipeline.rasterizer.lineWidth;
	}
	if(state.rasterizer.discard != pipeline.rasterizer.discard)
	{
		if(pipeline.rasterizer.discard)
			glEnable(GL_RASTERIZER_DISCARD);
		else
			glDisable(GL_RASTERIZER_DISCARD);
		state.rasterizer.discard = pipeline.rasterizer.discard;
	}
	if(state.rasterizer.colorWrite != pipeline.rasterizer.colorWrite)
	{
		glColorMask(pipeline.rasterizer.colorWrite, pipeline.rasterizer.colorWrite, pipeline.rasterizer.colorWrite, pipeline.rasterizer.colorWrite);
		state.rasterizer.colorWrite = pipeline.rasterizer.colorWrite;
	}
	if(state.rasterizer.dithering != pipeline.rasterizer.dithering)
	{
		if(pipeline.rasterizer.dithering)
			glEnable(GL_DITHER);
		else
			glDisable(GL_DITHER);
		state.rasterizer.dithering = pipeline.rasterizer.dithering;
	}

	// ***** Depth-stencil state **********************************************
	if(state.depthStencil.depthTest != pipeline.depthStencil.depthTest)
	{
		if(pipeline.depthStencil.depthTest)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);
		state.depthStencil.depthTest = pipeline.depthStencil.depthTest;
	}
	if(state.depthStencil.depthTest &&
		state.depthStencil.depthCmpFunc != pipeline.depthStencil.depthCmpFunc)
	{
		glDepthFunc(static_cast<GLenum>(pipeline.depthStencil.depthCmpFunc));
		state.depthStencil.depthCmpFunc = pipeline.depthStencil.depthCmpFunc;
	}
	if(state.depthStencil.depthWrite != pipeline.depthStencil.depthWrite)
	{
		glDepthMask(pipeline.depthStencil.depthWrite);
		state.depthStencil.depthWrite = pipeline.depthStencil.depthWrite;
	}
	if(state.depthStencil.stencilTest != pipeline.depthStencil.stencilTest)
	{
		if(pipeline.depthStencil.stencilTest)
			glEnable(GL_STENCIL_TEST);
		else
			glDisable(GL_STENCIL_TEST);
		state.depthStencil.stencilTest = pipeline.depthStencil.stencilTest;
	}
	if(state.depthStencil.stencilTest)
	{
		if(state.depthStencil.stencilCmpFuncFront != pipeline.depthStencil.stencilCmpFuncFront
			|| state.depthStencil.stencilRefFront != pipeline.depthStencil.stencilRefFront)
		{
			glStencilFuncSeparate(GL_FRONT, static_cast<GLenum>(pipeline.depthStencil.stencilCmpFuncFront), pipeline.depthStencil.stencilRefFront, 0xffffffff);
			state.depthStencil.stencilCmpFuncFront = pipeline.depthStencil.stencilCmpFuncFront;
			state.depthStencil.stencilRefFront = pipeline.depthStencil.stencilRefFront;
		}
		if(state.depthStencil.stencilCmpFuncBack != pipeline.depthStencil.stencilCmpFuncBack
			|| state.depthStencil.stencilRefBack != pipeline.depthStencil.stencilRefBack)
		{
			glStencilFuncSeparate(GL_BACK, static_cast<GLenum>(pipeline.depthStencil.stencilCmpFuncBack), pipeline.depthStencil.stencilRefBack, 0xffffffff);
			state.depthStencil.stencilCmpFuncBack = pipeline.depthStencil.stencilCmpFuncBack;
			state.depthStencil.stencilRefBack = pipeline.depthStencil.stencilRefBack;
		}
		if(state.depthStencil.stencilFailOpFront != pipeline.depthStencil.stencilFailOpFront
			|| state.depthStencil.zfailOpFront != pipeline.depthStencil.zfailOpFront
			|| state.depthStencil.passOpFront != pipeline.depthStencil.passOpFront)
		{
			glStencilOpSeparate(GL_FRONT, static_cast<GLenum>(pipeline.depthStencil.stencilFailOpFront),
				static_cast<GLenum>(pipeline.depthStencil.zfailOpFront),
				static_cast<GLenum>(pipeline.depthStencil.passOpFront));
			state.depthStencil.stencilFailOpFront = pipeline.depthStencil.stencilFailOpFront;
			state.depthStencil.zfailOpFront = pipeline.depthStencil.zfailOpFront;
			state.depthStencil.passOpFront = pipeline.depthStencil.passOpFront;
		}
		if(state.depthStencil.stencilFailOpBack != pipeline.depthStencil.stencilFailOpBack
			|| state.depthStencil.zfailOpBack != pipeline.depthStencil.zfailOpBack
			|| state.depthStencil.passOpBack != pipeline.depthStencil.passOpBack)
		{
			glStencilOpSeparate(GL_BACK, static_cast<GLenum>(pipeline.depthStencil.stencilFailOpBack),
				static_cast<GLenum>(pipeline.depthStencil.zfailOpBack),
				static_cast<GLenum>(pipeline.depthStencil.passOpBack));
			state.depthStencil.stencilFailOpBack = pipeline.depthStencil.stencilFailOpBack;
			state.depthStencil.zfailOpBack = pipeline.depthStencil.zfailOpBack;
			state.depthStencil.passOpBack = pipeline.depthStencil.passOpBack;
		}
	}
    if(state.depthStencil.polygonOffsetFactor != pipeline.depthStencil.polygonOffsetFactor ||
		state.depthStencil.polygonOffsetUnits != pipeline.depthStencil.polygonOffsetUnits || 
		state.depthStencil.polygonOffsetClamp != pipeline.depthStencil.polygonOffsetClamp) {
       if(pipeline.depthStencil.polygonOffsetFactor != 0.0f ||
		   pipeline.depthStencil.polygonOffsetUnits != 0.0f ||
		   pipeline.depthStencil.polygonOffsetClamp != 0.0f) {
		   glEnable(GL_POLYGON_OFFSET_FILL);
		   glEnable(GL_POLYGON_OFFSET_POINT);
		   glEnable(GL_POLYGON_OFFSET_LINE);
		   glPolygonOffsetClamp(pipeline.depthStencil.polygonOffsetFactor, pipeline.depthStencil.polygonOffsetUnits, pipeline.depthStencil.polygonOffsetClamp);
       } else {
		   glDisable(GL_POLYGON_OFFSET_FILL);
		   glDisable(GL_POLYGON_OFFSET_POINT);
		   glDisable(GL_POLYGON_OFFSET_LINE);
	   }
		state.depthStencil.polygonOffsetFactor = pipeline.depthStencil.polygonOffsetFactor;
		state.depthStencil.polygonOffsetUnits = pipeline.depthStencil.polygonOffsetUnits;
		state.depthStencil.polygonOffsetClamp = pipeline.depthStencil.polygonOffsetClamp;
    }
    

	// ***** Blend state ******************************************************
	if(state.blend.enableBlending != pipeline.blend.enableBlending)
	{
		if(pipeline.blend.enableBlending == BlendMode::Blend) {
			glDisable(GL_COLOR_LOGIC_OP);
			glEnable(GL_BLEND);
		}
		else if(pipeline.blend.enableBlending == BlendMode::Logic)
			glEnable(GL_COLOR_LOGIC_OP);
		else {
			glDisable(GL_BLEND);
			glDisable(GL_COLOR_LOGIC_OP);
		}
		state.blend.enableBlending = pipeline.blend.enableBlending;
	}
	if(state.blend.enableBlending == BlendMode::Blend)
	{
		for(int i = 0; i < 8; ++i)
		{
			if(state.blend.renderTarget[i].colorBlendOp != pipeline.blend.renderTarget[i].colorBlendOp
				|| state.blend.renderTarget[i].alphaBlendOp != pipeline.blend.renderTarget[i].alphaBlendOp)
			{
				glBlendEquationSeparatei(i, static_cast<GLenum>(pipeline.blend.renderTarget[i].colorBlendOp), static_cast<GLenum>(pipeline.blend.renderTarget[i].alphaBlendOp));
				state.blend.renderTarget[i].colorBlendOp = pipeline.blend.renderTarget[i].colorBlendOp;
				state.blend.renderTarget[i].alphaBlendOp = pipeline.blend.renderTarget[i].alphaBlendOp;
			}
			if(state.blend.renderTarget[i].srcColorFactor != pipeline.blend.renderTarget[i].srcColorFactor
				|| state.blend.renderTarget[i].srcAlphaFactor != pipeline.blend.renderTarget[i].srcAlphaFactor
				|| state.blend.renderTarget[i].dstColorFactor != pipeline.blend.renderTarget[i].dstColorFactor
				|| state.blend.renderTarget[i].dstAlphaFactor != pipeline.blend.renderTarget[i].dstAlphaFactor)
			{
				glBlendFuncSeparatei(i,
					static_cast<GLenum>(pipeline.blend.renderTarget[i].srcColorFactor), static_cast<GLenum>(pipeline.blend.renderTarget[i].dstColorFactor),
					static_cast<GLenum>(pipeline.blend.renderTarget[i].srcAlphaFactor), static_cast<GLenum>(pipeline.blend.renderTarget[i].dstAlphaFactor));
				state.blend.renderTarget[i].srcColorFactor = pipeline.blend.renderTarget[i].srcColorFactor;
				state.blend.renderTarget[i].srcAlphaFactor = pipeline.blend.renderTarget[i].srcAlphaFactor;
				state.blend.renderTarget[i].dstColorFactor = pipeline.blend.renderTarget[i].dstColorFactor;
				state.blend.renderTarget[i].dstAlphaFactor = pipeline.blend.renderTarget[i].dstAlphaFactor;
			}
		}
	}
	if(state.blend.logicOp != pipeline.blend.logicOp)
	{
		glLogicOp(static_cast<GLenum>(pipeline.blend.logicOp));
		state.blend.logicOp = pipeline.blend.logicOp;
	}
	if(state.blend.alphaToCoverage != pipeline.blend.alphaToCoverage)
	{
		if(pipeline.blend.alphaToCoverage)
			glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		else
			glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		state.blend.alphaToCoverage = pipeline.blend.alphaToCoverage;
	}

	// ***** Shader program ***************************************************
	mAssert(pipeline.program);
    glUseProgram(pipeline.program);

	if(pipeline.vertexArray)
		glBindVertexArray(pipeline.vertexArray);
	else // bind empty vertex format
		glBindVertexArray(get().m_emptyVao);
}

void Context::enableDepthWrite()
{
	auto& state = get().m_state;
	if (!state.depthStencil.depthWrite) {
		state.depthStencil.depthWrite = true;
		glDepthMask(GL_TRUE);
	}
}

Context::Context() {
	glGenVertexArrays(1, &m_emptyVao);
	glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, m_state.patch.outer.data());
	glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, m_state.patch.inner.data());
}

Context& Context::get() {
	static Context c;
	return c;
}
}
