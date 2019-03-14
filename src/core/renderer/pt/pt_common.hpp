#pragma once

#include "core/renderer/path_util.hpp"

namespace mufflon { namespace renderer {

struct PtVertexExt {
	//scene::Direction excident;
	//AngularPdf pdf;
	AreaPdf incidentPdf;

	CUDA_FUNCTION void init(const PathVertex<PtVertexExt>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const float incidentCosine, const AreaPdf incidentPdf,
			  const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<PtVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const AngularPdf pdfF, const AngularPdf pdfB) {
		//excident = sample.excident;
		//pdf = sample.pdfF;
	}
};

using PtPathVertex = PathVertex<PtVertexExt>;

}} // namespace mufflon::renderer