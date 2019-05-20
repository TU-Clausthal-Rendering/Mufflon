#include "cpu_bpm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"

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


	CUDA_FUNCTION void init(const BpmPathVertex& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosine,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		if(thisVertex.previous() && thisVertex.previous()->is_hitable()) {
			// Compute as much as possible from the conversion factor.
			// At this point we do not know n and A for the photons. This quantities
			// are added in the kernel after the walk.
			this->prevConversionFactor = float(
				thisVertex.previous()->convert_pdf(thisVertex.get_type(), AngularPdf{1.0f},
				{ incident, incidentDistance * incidentDistance }).pdf );
			if(thisVertex.previous()->is_end_point())
				this->prevConversionFactor /= float(thisVertex.previous()->ext().incidentPdf);
		}
	}

	CUDA_FUNCTION void update(const BpmPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const BpmPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			// Replace forward PDF with backward PDF (move merge one into the direction of the path-start)
			float relToPrev = float(pdf.back) * prevConversionFactor / float(thisVertex.ext().incidentPdf);
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}
	}
};

// MIS weight for merges
float get_mis_weight(const BpmPathVertex& viewVertex, const math::PdfPair pdf,
					 const CpuBidirPhotonMapper::PhotonDesc& photon) {
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

} // namespace ::


CpuBidirPhotonMapper::CpuBidirPhotonMapper() {
}

void CpuBidirPhotonMapper::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU BPM iteration", ProfileLevel::LOW);

	float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);

	// First pass: Create one photon path per view path
	u64 photonSeed = m_rngs[0].next();
	int numPhotons = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(int i = 0; i < numPhotons; ++i) {
		this->trace_photon(i, numPhotons, photonSeed, currentMergeRadius);
	}

	// Second pass: trace view paths and merge
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
					 pixel, numPhotons, currentMergeRadius);
	}
}

void CpuBidirPhotonMapper::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
}

void CpuBidirPhotonMapper::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2_1 rndStart { m_rngs[idx].next(), m_rngs[idx].next() };
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	BpmPathVertex vertex[2];
	BpmPathVertex::create_light(&vertex[0], nullptr, p);
	math::Throughput throughput;
	float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

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
		m_photonMap.insert(vertex[currentV].get_position(),
			{ vertex[currentV].get_position(), vertex[currentV].ext().incidentPdf,
			  vertex[currentV].get_incident_direction(), pathLen,
			  throughput.weight / numPhotons, vertex[otherV].ext().prevRelativeProbabilitySum,
			  vertex[currentV].get_geometric_normal(), vertex[currentV].ext().prevConversionFactor });
	} while(pathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path
}

void CpuBidirPhotonMapper::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeAreaInv = 1.0f / (ei::PI * mergeRadiusSq);
	// Trace view path
	BpmPathVertex vertex[2];
	// Create a start for the path
	BpmPathVertex::create_camera(&vertex[0], nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	math::Throughput throughput;
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

		// Evaluate direct hit of area ligths and background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = vertex[currentV].get_emission(m_sceneDesc, vertex[1-currentV].get_position());
			if(emission.value != 0.0f && viewPathLen > 1) {
				float misWeight = get_mis_weight(vertex[currentV], emission.pdf, emission.emitPdf, numPhotons, ei::PI * mergeRadiusSq);
				emission.value *= misWeight;
			}
			mAssert(!isnan(emission.value.x));
			m_outputBuffer.contribute(coord, throughput, emission.value, vertex[currentV].get_position(),
									vertex[currentV].get_normal(), vertex[currentV].get_albedo());
		}
		if(vertex[currentV].is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = vertex[currentV].get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int pathLen = viewPathLen + photonIt->pathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photonIt->position - currentPos) < mergeRadiusSq) {
				// Radiance estimate
				Pixel tmpCoord;
				auto bsdf = vertex[currentV].evaluate(-photonIt->incident,
													  m_sceneDesc.media, tmpCoord, false,
													  &photonIt->geoNormal);
				float misWeight = get_mis_weight(vertex[currentV], bsdf.pdf, *photonIt);
				radiance += bsdf.value * photonIt->flux * (mergeAreaInv * misWeight);
			}
			++photonIt;
		}
		m_outputBuffer.contribute(coord, throughput, radiance, scene::Point{0.0f},
			scene::Direction{0.0f}, Spectrum{0.0f});
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuBidirPhotonMapper::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer