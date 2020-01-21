#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"

#define SIL_SS_PT_USE_OCTREE

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

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
	i32 shadowInstanceId = -1;
	i32 silhouetteVerticesFirst[2u] = { -1, -1 };
	i32 silhouetteVerticesSecond[2u] = { -1, -1 }; // Two arrays in case of split vertices
	float silhouetteRegionSize = -1.f;

	float neeWeightedIrradiance = 0.f;
	float otherNeeLuminance = 0.f;


	CUDA_FUNCTION void init(const PathVertex<SilVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const VertexSample& sample) {
		this->excident = excident;
		this->pdf = sample.pdf.forw;
		this->outCos = ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& prevVertex,
							  const PathVertex<SilVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float continuationPropability,
							  const Spectrum& transmission) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const Spectrum& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt