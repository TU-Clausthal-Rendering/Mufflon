#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"
#include "core/renderer/footprint.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

struct SilVertexExt {
	using VertexType = PathVertex<SilVertexExt>;

	AreaPdf incidentPdf;
	FootprintV0 footprint;

	CUDA_FUNCTION void init(const VertexType& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		const float sourceCount = 1.f;
		this->footprint.init(1.0f / (float(inAreaPdf) * sourceCount), 1.0f / (float(inDirPdf) * sourceCount), pChoice);
	}

	CUDA_FUNCTION void update(const VertexType& prevVertex,
							  const VertexType& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float continuationPropability,
							  const Spectrum& transmission,
							  const scene::SceneDescriptor<CURRENT_DEV>& scene) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);

		this->footprint = prevVertex.ext().footprint.add_segment(
			static_cast<float>(pdf.forw), prevVertex.is_orthographic(),
			0.f, 0.f, 0.f, 1.f, incident.distance, 0.f, 1.0f);
	}

	CUDA_FUNCTION void update(const VertexType& thisVertex,
							  const scene::Direction& excident,
							  const VertexSample& sample,
							  const scene::SceneDescriptor<CURRENT_DEV>& scene) {}
};

using SilPathVertex = PathVertex<SilVertexExt>;

struct ShadowStatus {
	float shadow = 0.f;
	float light = 0.f;
	float neeBrightnes = 0.f;
};

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss