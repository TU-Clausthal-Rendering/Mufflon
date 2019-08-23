#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

// TODO: do border vertices get reduced? I don't think so
struct SilhouetteEdge {
	scene::PrimitiveHandle hitId{};
	float weight = 0;
};

template < Device dev >
struct Importances {
	cuda::Atomic<dev, float> viewImportance;	// Importance hits (not light!); also holds final normalized importance value after update
	cuda::Atomic<dev, float> irradiance;		// Accumulated irradiance
	cuda::Atomic<dev, u32> hitCounter;			// Number of hits
};

template < Device dev >
struct DeviceImportanceSums {
	cuda::Atomic<dev, float> shadowImportance;
	cuda::Atomic<dev, float> shadowSilhouetteImportance;
	cuda::Atomic<dev, u32> numSilhouettePixels;
};

struct ImportanceSums {
	float shadowImportance;
	float shadowSilhouetteImportance;
};

struct SilVertexExt {
	scene::Direction excident;
	AngularPdf pdf;
	AreaPdf incidentPdf;
	ei::Vec3 throughput;
	ei::Vec3 accumThroughput;
	float outCos;
	ei::Vec3 pathRadiance;
	float silhouetteRegionSize = 0.f;


	CUDA_FUNCTION void init(const PathVertex<SilVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		this->excident = excident;
		this->pdf = pdf.forw;
		this->outCos = ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& prevVertex,
							  const PathVertex<SilVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const math::Throughput& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum.weight;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss