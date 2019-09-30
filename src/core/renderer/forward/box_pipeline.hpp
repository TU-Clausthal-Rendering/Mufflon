#pragma once
#include "core/opengl/gl_object.h"
#include "core/opengl/gl_pipeline.h"
#include "core/memory/residency.hpp"
#include "ei/3dtypes.hpp"

namespace mufflon::renderer {
	
// helper class to draw bounding boxes with transparency
class BoxPipeline {
public:
	BoxPipeline();
	void init(gl::Framebuffer& framebuffer);
	void draw(gl::Handle box, uint32_t numBoxes, bool countingPass, const ei::Vec3& color) const;
	void draw(
		gl::Handle box,
		const ArrayDevHandle_t<Device::OPENGL, ei::Mat3x4>& transforms,
		uint32_t numBoxes, bool countingPass, const ei::Vec3& color) const;
	void draw(gl::Handle box, const ArrayDevHandle_t<Device::OPENGL, ei::Mat3x4>& transforms, 
		uint32_t instanceId, uint32_t numBoxes, bool countingPass, const ei::Vec3& color) const ;
private:
	gl::Program m_countProgram;
	gl::Program m_colorProgram;
	gl::Pipeline m_countPipe;
	gl::Pipeline m_colorPipe;
	gl::VertexArray m_vao;
};
}
