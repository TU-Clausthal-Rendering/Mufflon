#pragma once

#include "core/renderer/path_util.hpp"
#include <ei/vector.hpp>

namespace mufflon::renderer::decimaters::importance {

struct ImpVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;
	ei::Vec3 accumThroughput;
	float outCos;
	ei::Vec3 pathRadiance;


	CUDA_FUNCTION void init(const PathVertex<ImpVertexExt>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosineAbs,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<ImpVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		this->excident = excident;
		this->pdf = pdf.forw;
		this->outCos = -ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const math::Throughput& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum.weight;
	}
};

using ImpPathVertex = PathVertex<ImpVertexExt>;

} // namespace mufflon::renderer::decimaters::importance