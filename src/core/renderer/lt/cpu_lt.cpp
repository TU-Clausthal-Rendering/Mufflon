#include "cpu_lt.hpp"
#include "lt_common.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/parameter.hpp"

namespace mufflon::renderer {

CpuLightTracer::CpuLightTracer() {
	// The PT does not need additional memory resources like photon maps.
	logInfo("[CpuLightTracer] Size of a vertex is ", sizeof(LtPathVertex));
}

void CpuLightTracer::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU LT iteration", ProfileLevel::HIGH);

#pragma PARALLEL_FOR
	for(int photon = 0; photon < m_outputBuffer.get_num_pixels(); ++photon) {
		lt_sample(m_outputBuffer, m_sceneDesc, m_params, photon, m_rngs[photon]);
	}
}

void CpuLightTracer::post_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	logInfo("[CpuLightTracer] Params: path length in [", m_params.minPathLength, ", ",
		m_params.maxPathLength, "]");
}

void CpuLightTracer::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer