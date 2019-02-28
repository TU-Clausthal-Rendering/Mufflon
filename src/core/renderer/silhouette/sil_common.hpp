#pragma once

#include "core/renderer/path_util.hpp"

namespace mufflon::renderer {

struct SilVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;

	CUDA_FUNCTION void init(const PathVertex<SilVertexExt>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const float incidentCosine, const AreaPdf incidentPdf) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& thisVertex,
							  const math::PathSample& sample) {
		excident = sample.excident;
		pdf = sample.pdfF;
		throughput = sample.throughput;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

} // namespace mufflon::renderer