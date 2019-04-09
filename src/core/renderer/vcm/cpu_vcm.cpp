#include "cpu_vcm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct VcmVertexExt {
	AreaPdf incidentPdf;
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor { 0.0f };


	CUDA_FUNCTION void init(const VcmPathVertex& thisVertex,
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

	CUDA_FUNCTION void update(const VcmPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const VcmPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			// Replace forward PDF with backward PDF (move merge one into the direction of the path-start)
			float relToPrev = float(pdf.back) * prevConversionFactor / float(thisVertex.ext().incidentPdf);
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}
	}
};

// MIS weight for merges
float get_mis_weight(const VcmPathVertex& viewVertex, const math::PdfPair pdf,
					 const VcmPathVertex& photon) {
	// Add the merge at the previous view path vertex
	mAssert(viewVertex.previous() != nullptr);
	float relPdf = viewVertex.ext().prevConversionFactor * float(pdf.back)
			/ float(viewVertex.ext().incidentPdf);
	float otherProbSum = relPdf + relPdf * viewVertex.previous()->ext().prevRelativeProbabilitySum;
	// Add the merge or hit at the previous light path vertex
	AreaPdf nextLightPdf { float(pdf.forw) * photon.ext().prevConversionFactor };
	relPdf = nextLightPdf / photon.ext().incidentPdf;
	otherProbSum += relPdf + relPdf * photon.ext().prevRelativeProbabilitySum; // TODO: was prevPrev
	return 1.0f / (1.0f + otherProbSum);
}

// MIS weight for unidirectional hits.
float get_mis_weight(const VcmPathVertex& thisVertex, const AngularPdf pdfBack,
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


CpuVcm::CpuVcm() {}
CpuVcm::~CpuVcm() {}

void CpuVcm::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU VCM iteration", ProfileLevel::LOW);

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

	Profiler::instance().create_snapshot_all();
}

void CpuVcm::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
	m_pathEndPoints.resize(m_outputBuffer.get_num_pixels());
}

void CpuVcm::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2_1 rndStart { m_rngs[idx].next(), m_rngs[idx].next() };
	scene::lights::Photon p = emit(m_sceneDesc.lightTree, idx, numPhotons, seed,
		m_sceneDesc.aabb, rndStart);
	VcmPathVertex vertex;
	VcmPathVertex::create_light(&vertex, nullptr, p, m_rngs[idx]);	// TODO: check why there is an (unused) Rng reference
	const VcmPathVertex* previous = m_photonMap.insert(p.pos.position, vertex);
	math::Throughput throughput;
	float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

	int pathLen = 0;
	do {
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(!walk(m_sceneDesc, *previous, rnd, -1.0f, true, throughput, vertex, sample))
			break;
		++pathLen;

		// Complete the convertion factor with the quantities which where not known
		// to ext().init().
		if(pathLen == 1)
			vertex.ext().prevConversionFactor /= numPhotons * mergeArea;

		// Store a photon to the photon map
		previous = m_photonMap.insert(vertex.get_position(), vertex);
	} while(pathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path

	m_pathEndPoints[idx] = previous;
}

void CpuVcm::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeAreaInv = 1.0f / (ei::PI * mergeRadiusSq);
	// Trace view path
	VcmPathVertex vertex[2];
	// Create a start for the path
	VcmPathVertex::create_camera(&vertex[0], &vertex[0], m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	math::Throughput throughput;
	int currentV = 0;
	int viewPathLen = 0;
	do {
		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(!walk(m_sceneDesc, vertex[currentV], rnd, -1.0f, false, throughput, vertex[otherV], sample)) {
			if(throughput.weight != Spectrum{ 0.0f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					// For direction/envmap sources the sampling of position and direction is
					// reverted, so we need to cast the swapped pdfs to fit the expected order of events.
					AngularPdf backtracePdf = AngularPdf{ 1.0f / math::projected_area(sample.excident, m_sceneDesc.aabb) };
					AreaPdf startPdf = background_pdf(m_sceneDesc.lightTree, background);
					float misWeight = get_mis_weight(vertex[otherV], backtracePdf, startPdf, numPhotons, ei::PI * mergeRadiusSq);
					background.value *= misWeight;
					m_outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}
		++viewPathLen;
		currentV = otherV;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = vertex[currentV].get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int pathLen = viewPathLen + photonIt->get_path_len();
			if(photonIt->get_path_len() > 0 && pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photonIt->get_position() - currentPos) < mergeRadiusSq) {
				// Radiance estimate
				Pixel tmpCoord;
				scene::Direction geoNormal = photonIt->get_geometric_normal();
				auto bsdf = vertex[currentV].evaluate(-photonIt->get_incident_direction(),
													  m_sceneDesc.media, tmpCoord, false,
													  &geoNormal);
				float misWeight = get_mis_weight(vertex[currentV], bsdf.pdf, *photonIt);
				//radiance += bsdf.value * photonIt->flux * (mergeAreaInv * misWeight);
			}
			++photonIt;
		}
		m_outputBuffer.contribute(coord, throughput, radiance, scene::Point{0.0f},
			scene::Direction{0.0f}, Spectrum{0.0f});

		// Evaluate direct hit of area ligths
		if(viewPathLen >= m_params.minPathLength) {
			math::SampleValue emission = vertex[currentV].get_emission();
			if(emission.value != 0.0f && viewPathLen > 1) {
				AreaPdf startPdf = emit_pdf(m_sceneDesc.lightTree, vertex[currentV].get_primitive_id(),
											vertex[currentV].get_surface_params(), vertex[1-currentV].get_position(),
											scene::lights::guide_flux);
				float misWeight = get_mis_weight(vertex[currentV], emission.pdf, startPdf, numPhotons, ei::PI * mergeRadiusSq);
				emission.value *= misWeight;
			}
			mAssert(!isnan(emission.value.x));
			m_outputBuffer.contribute(coord, throughput, emission.value, vertex[currentV].get_position(),
									vertex[currentV].get_normal(), vertex[currentV].get_albedo());
		}
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuVcm::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer