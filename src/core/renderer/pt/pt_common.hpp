#pragma once

#include "core/renderer/path_util.hpp"

namespace mufflon { namespace renderer {

struct PtVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;

	CUDA_FUNCTION void init(const PathVertex<PtVertexExt, 4>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const float incidentCosine, const AreaPdf incidentPdf) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<PtVertexExt, 4>& thisVertex,
				const math::PathSample& sample) {
		excident = sample.excident;
		pdf = sample.pdfF;
	}
};

using PtPathVertex = PathVertex<PtVertexExt, 4>;

}} // namespace mufflon::renderer