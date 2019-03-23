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
#include "core/scene/lights/light_tree_sampling.hpp"

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

	math::Throughput throughput;	// Throughput of the path up to this point

	CUDA_FUNCTION void init(const BpmPathVertex& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput;
	}

	CUDA_FUNCTION void update(const BpmPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		/*const BptPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			AreaPdf prevReversePdf = prev->convert_pdf(thisVertex.get_type(), pdf.back,
				{thisVertex.get_incident_direction(), thisVertex.get_incident_dist_sq()});
			// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
			float relToPrev = prevReversePdf / prev->ext().incidentPdf;
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}*/
	}
};

} // namespace ::


CpuBidirPhotonMapper::CpuBidirPhotonMapper() {
}

void CpuBidirPhotonMapper::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU BPM iteration", ProfileLevel::LOW);

	float sceneSize = len(m_sceneDesc.aabb.max - m_sceneDesc.aabb.min);
	float currentMergeRadius = m_params.mergeRadius * sceneSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);

	// First pass: Create one photon path per view path
	u64 photonSeed = m_rngs[0].next();
#pragma PARALLEL_FOR
	for(int i = 0; i < m_outputBuffer.get_num_pixels(); ++i) {
		this->trace_photon(i, m_outputBuffer.get_num_pixels(), photonSeed);
	}

	// Second pass: trace view paths and merge
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
					 pixel, currentMergeRadius);
	}

	Profiler::instance().create_snapshot_all();
}

void CpuBidirPhotonMapper::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
}

void CpuBidirPhotonMapper::trace_photon(int idx, int numPhotons, u64 seed) {
	math::RndSet2_1 rndStart { m_rngs[idx].next(), m_rngs[idx].next() };
	scene::lights::Photon p = emit(m_sceneDesc.lightTree, idx, numPhotons, seed,
		m_sceneDesc.aabb, rndStart);
	BpmPathVertex vertex[2];
	BpmPathVertex::create_light(&vertex[0], &vertex[1], p, m_rngs[idx]);	// TODO: check why there is an (unused) Rng reference
	math::Throughput throughput;

	int pathLen = 0;
	int currentV = 0;
	do {
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		VertexSample sample;
		int otherV = 1 - currentV;
		if(!walk(m_sceneDesc, vertex[currentV], rnd, -1.0f, true, throughput, vertex[otherV], sample))
			break;
		++pathLen;
		currentV = otherV;

		// Store a photon to the photon map
		m_photonMap.insert(vertex[currentV].get_position(),
			{ vertex[currentV].get_position(), vertex[currentV].ext().incidentPdf,
			  vertex[currentV].get_incident_direction(), pathLen,
			  throughput.weight / numPhotons });
	} while(pathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path
}

void CpuBidirPhotonMapper::sample(const Pixel coord, int idx, float currentMergeRadius) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeAreaInv = 1.0f / (ei::PI * mergeRadiusSq);
	// Trace view path
	BpmPathVertex vertex[2];
	// Create a start for the path
	BpmPathVertex::create_camera(&vertex[0], &vertex[0], m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	math::Throughput throughput;
	int currentV = 0;
	int viewPathLen = 0;
	do {
		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		VertexSample sample;
		if(!walk(m_sceneDesc, vertex[currentV], rnd, -1.0f, false, throughput, vertex[otherV], sample)) {
			if(throughput.weight != Spectrum{ 0.0f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					// Update MIS for the last connection and that before.
					// For direction/envmap sources the sampling of position and direction is
					// reverted, so we need to cast the swapped pdfs to fit the expected order of events.
					//AngularPdf backtracePdf = AngularPdf{ 1.0f / math::projected_area(sample.excident, m_sceneDesc.aabb) };
					//AreaPdf startPdf = background_pdf(m_sceneDesc.lightTree, background);
					//float mis = get_mis_weight(vertex[otherV], backtracePdf, startPdf);
					//background.value *= mis;
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
			int pathLen = viewPathLen + photonIt->pathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photonIt->position - currentPos) < mergeRadiusSq) {
				// Radiance estimate
				Pixel tmpCoord;
				auto bsdf = vertex[currentV].evaluate(-photonIt->incident,
													  m_sceneDesc.media, tmpCoord, false, true);
				radiance += bsdf.value * photonIt->flux * mergeAreaInv;
			}
			++photonIt;
		}
		m_outputBuffer.contribute(coord, throughput, radiance, scene::Point{0.0f},
			scene::Direction{0.0f}, Spectrum{0.0f});

		// Evaluate direct hit of area ligths
		if(viewPathLen >= m_params.minPathLength) {
			math::SampleValue emission = vertex[currentV].get_emission();
			/*if(emission.value != 0.0f) {
				AreaPdf startPdf = emit_pdf(m_sceneDesc.lightTree, vertex[currentV].get_primitive_id(),
											vertex[currentV].get_surface_params(), vertex[1-currentV].get_position(),
											scene::lights::guide_flux);
				float mis = get_mis_weight(vertex[currentV], emission.pdf, startPdf);
				emission.value *= mis;
			}*/
			mAssert(!isnan(emission.value.x));
	//		m_outputBuffer.contribute(coord, throughput, emission.value, vertex[currentV].get_position(),
	//								vertex[currentV].get_normal(), vertex[currentV].get_albedo());
		}
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuBidirPhotonMapper::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer