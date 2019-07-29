#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"

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


	CUDA_FUNCTION void init(const PathVertex<SilVertexExt>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosineAbs,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		this->excident = excident;
		this->pdf = pdf.forw;
		this->outCos = ei::dot(thisVertex.get_normal(), excident);
	}

	CUDA_FUNCTION void updateBxdf(const VertexSample& sample, const math::Throughput& accum) {
		this->throughput = sample.throughput;
		this->accumThroughput = accum.weight;
	}
};

using SilPathVertex = PathVertex<SilVertexExt>;

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt