#pragma once

#include "core/renderer/path_util.hpp"
#include <ei/vector.hpp>

namespace mufflon::renderer::importance {

struct ImpVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;
	ei::Vec3 accumThroughput;
	float outCos;
	ei::Vec3 bxdfPdf;
	ei::Vec3 pathRadiance;

	CUDA_FUNCTION void init(const PathVertex<ImpVertexExt>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const float incidentCosine, const AreaPdf incidentPdf) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<ImpVertexExt>& thisVertex,
							  const math::PathSample& sample) {
		excident = sample.excident;
		pdf = sample.pdfF;
		throughput = sample.throughput;
		outCos = -ei::dot(thisVertex.get_normal(), sample.excident);
		bxdfPdf = sample.throughput / outCos;
	}
};

using ImpPathVertex = PathVertex<ImpVertexExt>;

} // namespace mufflon::renderer::importance