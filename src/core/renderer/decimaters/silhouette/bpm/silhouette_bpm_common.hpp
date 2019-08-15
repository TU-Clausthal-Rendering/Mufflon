#pragma once

#include "core/renderer/path_util.hpp"
#include "core/cuda/cuda_utils.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace bpm {

template < Device dev >
struct Importances {
	cuda::Atomic<dev, float> fluxImportance;
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

// Information which are stored in the photon map
struct PhotonDesc {
	scene::Point position;
	AreaPdf incidentPdf;
	scene::Direction incident;
	int pathLen;
	Spectrum flux;
	float prevPrevRelativeProbabilitySum;	// Sum of relative probabilities for merges and the connection up to the second previous vertex.
	scene::Direction geoNormal;				// Geometric normal at photon hit point. This is crucial for normal correction.
	float prevConversionFactor;				// 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	// TODO: improve memory footprint!
	PhotonDesc* prevPhoton;
	scene::PrimitiveHandle hitId;
};

struct SilVertexExt;
using SilPathVertex = PathVertex<SilVertexExt>;

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct SilVertexExt {
	AreaPdf incidentPdf;
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor{ 0.0f };


	CUDA_FUNCTION void init(const PathVertex<SilVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	CUDA_FUNCTION void update(const PathVertex<SilVertexExt>& prevVertex,
							  const PathVertex<SilVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
		if(prevVertex.is_hitable()) {
			// Compute as much as possible from the conversion factor.
			// At this point we do not know n and A for the photons. This quantities
			// are added in the kernel after the walk.
			float outCosAbs = ei::abs(prevVertex.get_geometric_factor(incident.dir));
			this->prevConversionFactor = orthoConnection ? outCosAbs : outCosAbs / incident.distanceSq;
			if(prevVertex.is_end_point())
				this->prevConversionFactor /= float(prevVertex.ext().incidentPdf);
		}
	}

	CUDA_FUNCTION void update(const SilPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const SilPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			// Replace forward PDF with backward PDF (move merge one into the direction of the path-start)
			float relToPrev = float(pdf.back) * prevConversionFactor / float(thisVertex.ext().incidentPdf);
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}
	}
};

}}}}} // namespace mufflon::renderer::decimaters::silhouette::bpm