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
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	CUDA_FUNCTION void update(const PathVertex<ImpVertexExt>& prevVertex,
							  const PathVertex<ImpVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {}

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