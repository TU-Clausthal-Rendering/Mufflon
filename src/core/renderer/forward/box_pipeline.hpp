#pragma once
#include "core/opengl/gl_object.hpp"
#include "core/opengl/gl_pipeline.hpp"
#include "core/memory/residency.hpp"
#include "ei/3dtypes.hpp"

namespace mufflon::renderer {
	
// helper class to draw bounding boxes with transparency
class BoxPipeline {
public:
	BoxPipeline();
	void init(gl::Framebuffer& framebuffer);
	void draw(gl::Handle box, gl::Handle levels, int numBoxes, int numLevel, bool countingPass, const ei::Vec3& color) const;
	void draw(gl::Handle box, gl::Handle levels, ei::Mat3x4 transforms, int numBoxes, int numLevel, bool countingPass, const ei::Vec3& color) const;

	void draw(gl::Handle box, const ArrayDevHandle_t<Device::OPENGL, ei::Mat3x4>& transforms, uint32_t numBoxes, bool countingPass, const ei::Vec3& color) const ;
private:
	// default count pipe
	gl::Pipeline m_countPipe;
	gl::Pipeline m_colorPipe;
	gl::Program m_countProgram;
	gl::Program m_colorProgram;	
	gl::VertexArray m_vao;
	// extended to include box hierarchy level
	gl::Pipeline m_countPipeEx;
	gl::Pipeline m_colorPipeEx;
	gl::Program m_countProgramEx;
	gl::Program m_colorProgramEx;
	gl::VertexArray m_vaoExt;
};
}
