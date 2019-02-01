#pragma once

#include "cpu_pt.hpp"
#include "gpu_pt.hpp"
#include "silhouette/silhouette.hpp"
#include "silhouette/wireframe.hpp"
#include "util/tagged_tuple.hpp"

namespace mufflon::renderer {

using Renderers = util::TaggedTuple<CpuPathTracer, GpuPathTracer, silhouette::WireframeRenderer>;
//using Renderers = util::TaggedTuple<CpuPathTracer, GpuPathTracer, silhouette::WireframeRenderer,	silhouette::SilhouetteTracer>;

} // namespace mufflon::renderer