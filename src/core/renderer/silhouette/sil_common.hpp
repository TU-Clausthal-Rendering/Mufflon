#pragma once

#include "core/renderer/path_util.hpp"

namespace mufflon::renderer {

struct SilVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;
	ei::Vec3 accumThroughput;
	float outCos;
	ei::Vec3 bxdfPdf;
	ei::Vec3 pathRadiance;
	ei::Ray shadowRay;
	float lightDistance;
	scene::PrimitiveHandle shadowHit;
	float firstShadowDistance;

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
		outCos = -ei::dot(thisVertex.get_normal(), sample.excident);
		bxdfPdf = sample.throughput / outCos;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

} // namespace mufflon::renderer