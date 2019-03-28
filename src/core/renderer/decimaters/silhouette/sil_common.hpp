#pragma once

#include "core/renderer/path_util.hpp"

namespace mufflon::renderer::decimaters::silhouette {

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
							const float incidentCosine, const AreaPdf incidentPdf,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const AngularPdf pdfF, const AngularPdf pdfB) {
		this->excident = excident;
		this->pdf = pdfF;
		this->outCos = -ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const math::Throughput& accum) {
		this->throughput = sample.throughput;
		this->bxdfPdf = this->throughput / this->outCos;
		this->accumThroughput = accum.weight;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

} // namespace mufflon::renderer::decimaters::silhouette