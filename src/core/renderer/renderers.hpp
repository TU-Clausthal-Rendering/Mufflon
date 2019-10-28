#pragma once

#include "util/tagged_tuple.hpp"
#include "core/renderer/pt/cpu_pt.hpp"
#include "core/renderer/pt/gpu_pt.hpp"
#include "core/renderer/pt/hybrid_pt.hpp"
#include "core/renderer/decimaters/importance/cpu_importance.hpp"
#include "core/renderer/decimaters/silhouette/pt/cpu_silhouette_pt.hpp"
#include "core/renderer/decimaters/silhouette/pt/gpu_silhouette_pt.hpp"
#include "core/renderer/decimaters/silhouette/screenspace/cpu_ss_sil_pt.hpp"
#include "core/renderer/decimaters/silhouette/screenspace/gpu_ss_sil_pt.hpp"
#include "core/renderer/decimaters/silhouette/bpm/cpu_silhouette_bpm.hpp"
#include "core/renderer/decimaters/silhouette/bpm/gpu_silhouette_bpm.hpp"
#include "core/renderer/decimaters/shadow_photons/cpu_shadow_photons.hpp"
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
	decimaters::silhouette::CpuShadowSilhouettesPT, decimaters::silhouette::CpuSsSilPT, decimaters::silhouette::GpuShadowSilhouettesPT,
	decimaters::silhouette::CpuShadowSilhouettesBPM,
	decimaters::CpuImportanceDecimater,
	decimaters::spm::ShadowPhotonVisualizer
>;

} // namespace mufflon::renderer
