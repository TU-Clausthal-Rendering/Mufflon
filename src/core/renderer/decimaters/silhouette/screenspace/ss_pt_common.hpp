#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

struct SilVertexExt {
	using VertexType = PathVertex<SilVertexExt>;

	AreaPdf incidentPdf;

	CUDA_FUNCTION void init(const VertexType& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	CUDA_FUNCTION void update(const VertexType& prevVertex,
							  const VertexType& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
	}

	CUDA_FUNCTION void update(const VertexType& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {}
};

using SilPathVertex = PathVertex<SilVertexExt>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss