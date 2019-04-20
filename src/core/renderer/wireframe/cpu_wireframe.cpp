#include "cpu_wireframe.hpp"
#include "wireframe_common.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include <random>

namespace mufflon::renderer {

using PtPathVertex = PathVertex<VertexExtension>;

void CpuWireframe::post_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());
}

void CpuWireframe::iterate() {
	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() });
	}
}

void CpuWireframe::sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();
	sample_wireframe(m_outputBuffer, m_sceneDesc, m_params, m_rngs[pixel], coord);
}

void CpuWireframe::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer