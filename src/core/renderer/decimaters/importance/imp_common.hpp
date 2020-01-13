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
							  const Spectrum& throughput,
							  const float continuationPropability,
							  const Spectrum& transmission) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
	}

	CUDA_FUNCTION void update(const PathVertex<ImpVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const VertexSample& sample) {
		this->excident = excident;
		this->pdf = sample.pdf.forw;
		this->outCos = ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const Spectrum& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum;
	}
};

using ImpPathVertex = PathVertex<ImpVertexExt>;

} // namespace mufflon::renderer::decimaters::importance