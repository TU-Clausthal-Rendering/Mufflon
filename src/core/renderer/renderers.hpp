#pragma once

#include "util/tagged_tuple.hpp"
#include "core/renderer/pt/cpu_pt.hpp"
#include "core/renderer/pt/gpu_pt.hpp"
#include "core/renderer/silhouette/silhouette.hpp"
#include "core/renderer/wireframe/cpu_wireframe.hpp"
#include "core/renderer/wireframe/gpu_wireframe.hpp"

namespace mufflon::renderer {

using Renderers = util::TaggedTuple<CpuPathTracer, GpuPathTracer, CpuWireframe, GpuWireframe>;

} // namespace mufflon::renderer