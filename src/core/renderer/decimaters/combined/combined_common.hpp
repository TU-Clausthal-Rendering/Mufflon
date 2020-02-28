#pragma once

#include "core/export/core_api.h"
#include "core/renderer/footprint.hpp"
#include "core/renderer/path_util.hpp"
#include "core/scene/types.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace combined {

struct ShadowStatus {
	float shadowContributions = 0.f;
	float shadow = 0.f;
	float light = 0.f;
};

struct CombinedVertexExt {
	FootprintV0 footprint;
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;
	ei::Vec3 accumThroughput;
	float outCos;
	ei::Vec3 pathRadiance;
	i32 shadowInstanceId = -1;
	i32 silhouetteVerticesFirst[2u] = { -1, -1 };
	i32 silhouetteVerticesSecond[2u] = { -1, -1 }; // Two arrays in case of split vertices
	float silhouetteRegionSize = -1.f;

	float neeWeightedIrradiance = 0.f;
	float otherNeeLuminance = 0.f;


	CUDA_FUNCTION void init(const PathVertex<CombinedVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		const float sourceCount = 1.f;
		this->footprint.init(1.0f / (float(inAreaPdf) * sourceCount), 1.0f / (float(inDirPdf) * sourceCount), pChoice);
	}

	CUDA_FUNCTION void update(const PathVertex<CombinedVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const VertexSample& sample) {
		this->excident = excident;
		this->pdf = sample.pdf.forw;
		this->outCos = ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void update(const PathVertex<CombinedVertexExt>& prevVertex,
							  const PathVertex<CombinedVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float continuationPropability,
							  const Spectrum& transmission) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
		this->footprint = prevVertex.ext().footprint.add_segment(
			static_cast<float>(pdf.forw), prevVertex.is_orthographic(),
			0.f, 0.f, 0.f, 1.f, incident.distance, 0.f, 1.0f);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const Spectrum& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum;
	}
};

using CombinedPathVertex = PathVertex<CombinedVertexExt>;

}}}} // namespace mufflon::renderer::decimaters::combined