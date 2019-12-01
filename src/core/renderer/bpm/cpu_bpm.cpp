#include "cpu_bpm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <cmath>

namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BpmVertexExt {
	AreaPdf incidentPdf;
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor { 0.0f };


	inline CUDA_FUNCTION void init(const BpmPathVertex& /*thisVertex*/,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	}

	inline CUDA_FUNCTION void update(const BpmPathVertex& prevVertex,
							  const BpmPathVertex& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& /*throughput*/,
							  const float /*continuationPropability*/,
							  const Spectrum& /*transmission*/) {
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

	inline CUDA_FUNCTION void update(const BpmPathVertex& thisVertex,
							  const scene::Direction& /*excident*/,
							  const VertexSample& sample) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const BpmPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			// Replace forward PDF with backward PDF (move merge one into the direction of the path-start)
			float relToPrev = float(sample.pdf.back) * prevConversionFactor / float(thisVertex.ext().incidentPdf);
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}
	}
};

// MIS weight for merges
float get_mis_weight(const BpmPathVertex& viewVertex, const math::PdfPair pdf,
					 const CpuBidirPhotonMapper::PhotonDescCommon& photon) {
	// Add the merge at the previous view path vertex
	mAssert(viewVertex.previous() != nullptr);
	float relPdf = viewVertex.ext().prevConversionFactor * float(pdf.back)
			/ float(viewVertex.ext().incidentPdf);
	float otherProbSum = relPdf + relPdf * viewVertex.previous()->ext().prevRelativeProbabilitySum;
	// Add the merge or hit at the previous light path vertex
	AreaPdf nextLightPdf { float(pdf.forw) * photon.prevConversionFactor };
	relPdf = nextLightPdf / photon.incidentPdf;
	otherProbSum += relPdf + relPdf * photon.prevPrevRelativeProbabilitySum;
	return 1.0f / (1.0f + otherProbSum);
}

// MIS weight for unidirectional hits.
float get_mis_weight(const BpmPathVertex& thisVertex, const AngularPdf pdfBack,
					 const AreaPdf startPdf, int numPhotons, float mergeArea) {
	mAssert(thisVertex.previous() != nullptr);
	// There is one merge which is not yet accounted for
	float relPdf = thisVertex.ext().prevConversionFactor * float(pdfBack)
			/ float(thisVertex.ext().incidentPdf);
	// Until now, merges where only compared to other merges. The reuse factor and the
	// merge area are not yet included.
	relPdf *= float(startPdf) * numPhotons * mergeArea;
	float mergeProbSum = relPdf + relPdf * thisVertex.previous()->ext().prevRelativeProbabilitySum;
	return 1.0f / (1.0f + mergeProbSum);
}

inline CUDA_FUNCTION float select_bandwidth(const float* distSq, int n) {
	// Take element k+1 for the independent area estimate (using the radius of the k-th element is biased).
	return distSq[n];	// Unbiased according to Garcia 2012, but biased in my experiments

	//return (distSq[n] + distSq[n-1]) / 2.0f;	// Good results in experiments for uniform
	//return distSq[n-1];
}

// A photon mapping kernel with dSq = current sample distance and rSq = bandwidth.
inline CUDA_FUNCTION float kernel(float dSq, float rSq) {
	//return 1.0f;	// Uniform
	//return 2.0f * (1.0f - dSq / rSq);		// Epanechnikov
	return 3.0f * ei::sq(1.0f - dSq / rSq);	// Silverman
}

} // namespace ::


CpuBidirPhotonMapper::CpuBidirPhotonMapper() {
}

void CpuBidirPhotonMapper::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU BPM iteration", ProfileLevel::LOW);

	float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);

	// Clear photons from previous iteration
	if(m_params.knn == 0)
		m_photonMap.clear(currentMergeRadius * 2.0001f);
	else
		m_photonMapKd.clear();

	// First pass: Create one photon path per view path
	u64 photonSeed = m_rngs[0].next();
	int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < numPhotons; ++i) {
		this->trace_photon(i, numPhotons, photonSeed, currentMergeRadius);
	}

	if(m_params.knn > 0) {
		m_photonMapKd.build();
		// For the MIS it is necessary to query the density at each photon.
		// Fortunatelly, the photons where added to the tree in forward order.
		// I.e. if we iterate NON-PARALLEL each previous photon is completed when the
		// next one is arrived.
		// 1. prallel query
#pragma PARALLEL_FOR
		for(int i = 0; i < m_photonMapKd.size(); ++i) {
			PhotonDescKNN& photon = m_photonMapKd.get_data_by_index(i);
			const ei::Vec3& currentPos = m_photonMapKd.get_position_by_index(i);
			int* indices = m_knnQueryMem.data() + get_current_thread_idx() * (m_params.knn+2) * 2;
			float* distSq = as<float>(indices + m_params.knn+2);
			m_photonMapKd.query_euclidean(currentPos, m_params.knn + 2, indices, distSq, ei::sq(currentMergeRadius));
			photon.mergeArea = select_bandwidth(distSq, m_params.knn+1) * ei::PI;
			mAssert(photon.mergeArea > 0.0f);
		}
		// 2. sequential update of perv... members
		for(int i = 0; i < m_photonMapKd.size(); ++i) {
			PhotonDescKNN& photon = m_photonMapKd.get_data_by_index(i);
			float prevArea = photon.previousIdx == -1 ? 1.0f
				: m_photonMapKd.get_data_by_index(photon.previousIdx).mergeArea;
			photon.prevConversionFactor *= prevArea / photon.mergeArea;
		}
	}

	// Second pass: trace view paths and merge
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
					 pixel, numPhotons, currentMergeRadius);
	}
}

void CpuBidirPhotonMapper::post_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	// Initilize one of two data structures and remove the other one
	if(m_params.knn == 0) {
		m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_photonMapKd.reserve(0);
	} else {
		m_photonMapManager.resize(0);
		m_photonMapKd.reserve(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
		m_knnQueryMem.resize((m_params.knn + 2) * 2 * get_thread_num());
	}
}

void CpuBidirPhotonMapper::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	BpmPathVertex vertex[2];
	BpmPathVertex::create_light(&vertex[0], nullptr, p);
	Spectrum throughput { 1.0f };
	float mergeArea = m_params.knn > 0 ? 1.0f :	// The merge area is dynamic in KNN searches
		ei::PI * currentMergeRadius * currentMergeRadius;
	int prevKdTreeIdx = -1;

	int pathLen = 0;
	int currentV = 0;
	int otherV = 1;
	do {
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette, true, throughput, vertex[otherV], sample) != WalkResult::HIT)
			break;
		++pathLen;
		currentV = otherV;
		otherV = 1 - currentV;
		// Complete the convertion factor with the quantities which where not known
		// to ext().init().
		if(pathLen == 1)
			vertex[currentV].ext().prevConversionFactor /= numPhotons * mergeArea;

		// Store a photon to the photon map
		if(m_params.knn == 0)
			m_photonMap.insert(vertex[currentV].get_position(), {
				vertex[currentV].ext().incidentPdf,
				vertex[currentV].get_incident_direction(), pathLen,
				throughput / numPhotons, vertex[otherV].ext().prevRelativeProbabilitySum,
				vertex[currentV].get_geometric_normal(), vertex[currentV].ext().prevConversionFactor,
				vertex[currentV].get_position() });
		else
			prevKdTreeIdx = m_photonMapKd.insert(vertex[currentV].get_position(), {
				vertex[currentV].ext().incidentPdf,
				vertex[currentV].get_incident_direction(), pathLen,
				throughput / numPhotons, vertex[otherV].ext().prevRelativeProbabilitySum,
				vertex[currentV].get_geometric_normal(), vertex[currentV].ext().prevConversionFactor,
				1.0f, prevKdTreeIdx });
	} while(pathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path
}

void CpuBidirPhotonMapper::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeAreaInv = 1.0f / (ei::PI * mergeRadiusSq);
	float prevMergeArea = 1.0f;
	// Trace view path
	BpmPathVertex vertex[2];
	// Create a start for the path
	BpmPathVertex::create_camera(&vertex[0], nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	Spectrum throughput { 1.0f };
	int currentV = 0;
	int viewPathLen = 0;
	do {
		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette, false, throughput, vertex[otherV], sample) == WalkResult::CANCEL)
			break;
		++viewPathLen;
		currentV = otherV;

		// Evaluate direct hit of area lights and background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = vertex[currentV].get_emission(m_sceneDesc, vertex[1-currentV].get_position());
			if(emission.value != 0.0f && viewPathLen > 1) {
				float misWeight = get_mis_weight(vertex[currentV], emission.pdf, emission.emitPdf, numPhotons, prevMergeArea);
				emission.value *= misWeight;
			}
			mAssert(!std::isnan(emission.value.x));
			m_outputBuffer.contribute<RadianceTarget>(coord, throughput * emission.value);
		}
		if(vertex[currentV].is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = vertex[currentV].get_position();
		if(m_params.knn == 0) {
			auto photonIt = m_photonMap.find_first(currentPos);
			while(photonIt) {
				// Only merge photons which are within the sphere around our position.
				// and which have the correct full path length.
				int pathLen = viewPathLen + photonIt->pathLen;
				if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
					&& lensq(photonIt->position - currentPos) < mergeRadiusSq) {
					radiance += merge(vertex[currentV], *photonIt);
					mAssert(!std::isnan(radiance.x));
				}
				++photonIt;
			}
			radiance *= mergeAreaInv;
			prevMergeArea = ei::PI * mergeRadiusSq;
		} else {
			int* indices = m_knnQueryMem.data() + get_current_thread_idx() * (m_params.knn + 1) * 2;
			float* distSq = as<float>(indices + m_params.knn+1);
			m_photonMapKd.query_euclidean(currentPos, m_params.knn + 1, indices, distSq, ei::sq(currentMergeRadius));
			float bandwidth = select_bandwidth(distSq, m_params.knn);
			float currentMergeArea = ei::PI * bandwidth;
			// Prepare view-path MIS differences due to area
			vertex[currentV].ext().prevConversionFactor *= prevMergeArea / currentMergeArea;
			prevMergeArea = currentMergeArea;
			for(int i = 0; i < m_params.knn && indices[i] >= 0; ++i) {
				const PhotonDescKNN& photon = m_photonMapKd.get_data_by_index(indices[i]);
				int pathLen = viewPathLen + photon.pathLen;
				if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength)
					radiance += merge(vertex[currentV], photon) * kernel(distSq[i], bandwidth);
			}
			radiance /= currentMergeArea;
			mAssert(!std::isnan(radiance.x));
		}

		m_outputBuffer.contribute<RadianceTarget>(coord, throughput * radiance);
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuBidirPhotonMapper::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

Spectrum CpuBidirPhotonMapper::merge(const BpmPathVertex& vertex, const PhotonDescCommon& photon) {
	// Radiance estimate
	Pixel tmpCoord;
	auto bsdf = vertex.evaluate(-photon.incident,
								 m_sceneDesc.media, tmpCoord, false,
								 &photon.geoNormal);
	float misWeight = get_mis_weight(vertex, bsdf.pdf, photon);
	return bsdf.value * photon.flux * misWeight;
}

} // namespace mufflon::renderer