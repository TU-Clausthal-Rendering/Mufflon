#include "cpu_pt.hpp"
#include "pt_common.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/parameter.hpp"

namespace mufflon::renderer {

CpuPathTracer::CpuPathTracer() {
	// The PT does not need additional memory resources like photon maps.
	logInfo("[CpuPathTracer] Size of a vertex is ", sizeof(PtPathVertex));
}

void CpuPathTracer::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU PT iteration", ProfileLevel::HIGH);

	m_sceneDesc.lightTree.posGuide = m_params.neeUsePositionGuide;

	// TODO: better pixel order?
	// TODO: different scheduling?
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		Pixel coord { pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		pt_sample(m_outputBuffer, m_sceneDesc, m_params, coord, m_rngs[pixel]);
	}
}

void CpuPathTracer::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	logInfo("[CpuPathTracer] Params: path length in [", m_params.minPathLength, ", ",
		m_params.maxPathLength, "]; nee count ", m_params.neeCount,
		"; position guide: ", m_params.neeUsePositionGuide);
}

void CpuPathTracer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer