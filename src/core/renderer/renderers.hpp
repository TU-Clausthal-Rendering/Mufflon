#pragma once

#include "util/tagged_tuple.hpp"
#include "core/renderer/pt/cpu_pt.hpp"
#include "core/renderer/pt/gpu_pt.hpp"
#include "core/renderer/decimaters/importance/cpu_importance.hpp"
#include "core/renderer/decimaters/silhouette/pt/cpu_silhouette_pt.hpp"
#include "core/renderer/decimaters/silhouette/pt/gpu_silhouette_pt.hpp"
#include "core/renderer/decimaters/silhouette/bpm/cpu_silhouette_bpm.hpp"
#include "core/renderer/decimaters/silhouette/bpm/gpu_silhouette_bpm.hpp"
#include "core/renderer/wireframe/cpu_wireframe.hpp"
#include "core/renderer/wireframe/gpu_wireframe.hpp"
#include "core/renderer/bpt/cpu_bpt.hpp"
#include "core/renderer/bpm/cpu_bpm.hpp"
#include "core/renderer/neb/cpu_neb.hpp"
#include "core/renderer/vcm/cpu_vcm.hpp"

namespace mufflon::renderer {

using Renderers = util::TaggedTuple<
	CpuPathTracer, GpuPathTracer,
	CpuWireframe, GpuWireframe,
	CpuBidirPathTracer,
	CpuBidirPhotonMapper,
	CpuNextEventBacktracking,
	CpuVcm,
	decimaters::silhouette::CpuShadowSilhouettesPT, decimaters::silhouette::GpuShadowSilhouettesPT,
	decimaters::silhouette::CpuShadowSilhouettesBPM,
	decimaters::CpuImportanceDecimater
>;

} // namespace mufflon::renderer
