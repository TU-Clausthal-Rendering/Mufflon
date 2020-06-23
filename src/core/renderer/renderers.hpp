#pragma once

#include "util/tagged_tuple.hpp"
#include "core/renderer/pt/cpu_pt.hpp"
#include "core/renderer/pt/gpu_pt.hpp"
#include "core/renderer/pt/hybrid_pt.hpp"
#include "core/renderer/decimaters/combined/cpu_combined_reducer.hpp"
#include "core/renderer/lt/cpu_lt.hpp"
#include "core/renderer/lt/gpu_lt.hpp"
#include "core/renderer/wireframe/cpu_wireframe.hpp"
#include "core/renderer/wireframe/gpu_wireframe.hpp"
#include "core/renderer/bpt/cpu_bpt.hpp"
#include "core/renderer/bpm/cpu_bpm.hpp"
#include "core/renderer/neb/cpu_neb.hpp"
#include "core/renderer/vcm/cpu_vcm.hpp"
#include "core/renderer/ivcm/cpu_ivcm.hpp"
#include "core/renderer/forward/gl_forward.hpp"
#include "core/renderer/wireframe/gl_wireframe.h"
#include "core/renderer/debug/debug_bvh_renderer.hpp"

#include "core/renderer/decimaters/plain/vertex_clusterer.hpp"

namespace mufflon::renderer {

using Renderers = util::TaggedTuple<
	CpuPathTracer, GpuPathTracer, HybridPathTracer,
	CpuLightTracer, GpuLightTracer,
	CpuWireframe, GpuWireframe, GlWireframe,
	CpuBidirPathTracer,
	CpuBidirPhotonMapper,
	CpuNextEventBacktracking,
	CpuVcm, CpuIvcm,
	GlForward, DebugBvhRenderer,
	decimaters::CpuCombinedReducer,

	decimaters::CpuUniformVertexClusterer, decimaters::GpuUniformVertexClusterer,
	decimaters::CpuOctreeVertexClusterer
>;

} // namespace mufflon::renderer
