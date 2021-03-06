﻿#include "cpu_vcm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

#include <cn/rnd.hpp>
#include <cmath>

namespace mufflon::renderer {

namespace {

// Compute the previous relative event sum in relation to a connection
// between the previous and the current vertex.
float prev_rel_sum(const VcmPathVertex& vertex, AngularPdf pdfBack,
	int numPhotons, float area) {
	const VcmPathVertex* prev = vertex.previous();
	if(prev) { // !prev: Current one is a start vertex. There is no previous sum
		// Replace forward PDF with backward PDF (move merge one into the direction of the path-start)
		float pdfBackA = float(pdfBack) * vertex.ext().prevConversionFactor;
		float relMerge = prev->previous() ? pdfBackA * numPhotons * area : 0.0f; // No merges at end points
		float relConn = pdfBackA / float(prev->ext().incidentPdf);
		return relConn + relMerge + relConn * prev->ext().prevRelativeProbabilitySum;
	}
	return 0.0f;
}

// MIS weight for merges
float get_mis_weight(const VcmPathVertex& viewVertex, const math::PdfPair pdf,
					 const VcmPathVertex& lightVertex, int numPhotons, float area) {
	// Complete previous merge and connection on view sub-path
	float relSumV = 1.0f + prev_rel_sum(viewVertex, pdf.back, numPhotons, area);
	// Complete previous merge and connection on light sub-path
	float relSumL = 1.0f + prev_rel_sum(lightVertex, pdf.forw, numPhotons, area);
	// Both sums are computed as if a connection from the previous to the current
	// vertex was made. The current merge differs by p_acc (p_in * A * n).
	relSumV /= float(viewVertex.ext().incidentPdf) * numPhotons * area;
	relSumL /= float(lightVertex.ext().incidentPdf) * numPhotons * area;
	return 1.0f / (1.0f + relSumV + relSumL);
}

// MIS weight for connections
float get_mis_weight(const VcmPathVertex& viewVertex, const math::PdfPair viewPdf,
					 const VcmPathVertex& lightVertex, const math::PdfPair lightPdf,
					 const Connection& connection, int numPhotons, float area) {
	// Complete previous merge and connection on view sub-path
	float relSumV = 1.0f + prev_rel_sum(viewVertex, viewPdf.back, numPhotons, area);
	// Complete previous merge and connection on light sub-path
	float relSumL = 1.0f + prev_rel_sum(lightVertex, lightPdf.back, numPhotons, area);
	// Both sums are computed as if a connection from the previous to the current
	// vertex was made.
	AreaPdf lightToViewPdf = viewVertex.convert_pdf(lightVertex.get_type(), lightPdf.forw, connection).pdf;
	relSumV *= lightToViewPdf / viewVertex.ext().incidentPdf;
	if(viewVertex.previous()) relSumV += float(lightToViewPdf) * numPhotons * area;
	AreaPdf viewToLightPdf = lightVertex.convert_pdf(viewVertex.get_type(), viewPdf.forw, connection).pdf;
	relSumL *= viewToLightPdf / lightVertex.ext().incidentPdf;
	if(lightVertex.previous()) relSumL += float(viewToLightPdf) * numPhotons * area;
	return 1.0f / (1.0f + relSumV + relSumL);
}

// MIS weight for unidirectional hits.
float get_mis_weight(const VcmPathVertex& thisVertex, const AngularPdf pdfBack,
					 const AreaPdf startPdf, int numPhotons, float area) {
	mAssert(thisVertex.previous() != nullptr);
	float relSum = 1.0f + prev_rel_sum(thisVertex, pdfBack, numPhotons, area);
	float relRndHit = startPdf / thisVertex.ext().incidentPdf;
	relSum *= relRndHit;
	return 1.0f / (1.0f + relSum);
}

struct ConnectionValue { Spectrum bxdfs; float cosines; };
ConnectionValue connect(const VcmPathVertex& path0, const VcmPathVertex& path1,
						const scene::SceneDescriptor<Device::CPU>& scene,
						Pixel& coord, int numPhotons, float area
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!VcmPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = VcmPathVertex::get_connection(path0, path1);
	auto val0 = path0.evaluate( connection.dir, scene.media, coord, false);
	auto val1 = path1.evaluate(-connection.dir, scene.media, coord, true);
	// Cancel reprojections outside the screen
	if(coord.x == -1) return {Spectrum{0.0f}, 0.0f};
	Spectrum bxdfProd = val0.value * val1.value;
	float cosProd = val0.cosOut * val1.cosOut;//TODO: abs?
	mAssert(cosProd >= 0.0f);
	mAssert(!std::isnan(bxdfProd.x));
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
		// Shadow test
		if(!scene::accel_struct::any_intersection(
				scene,connection.v0, path1.get_position(connection.v0),
				path0.get_geometric_normal(),  path1.get_geometric_normal(), connection.dir)) {
			bxdfProd *= scene.media[path0.get_medium(connection.dir)].get_transmission(connection.distance);
			float mis = get_mis_weight(path0, val0.pdf, path1, val1.pdf, connection, numPhotons, area);
			return {bxdfProd * (mis / connection.distanceSq), cosProd};
		}
	}
	return {Spectrum{0.0f}, 0.0f};
}

} // namespace ::

CUDA_FUNCTION void VcmVertexExt::init(const VcmPathVertex& /*thisVertex*/,
									  const AreaPdf inAreaPdf,
									  const AngularPdf inDirPdf,
									  const float pChoice) {
	this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	this->throughput = Spectrum{ 1.0f };
}

CUDA_FUNCTION void VcmVertexExt::update(const VcmPathVertex& prevVertex,
										const VcmPathVertex& thisVertex,
										const math::PdfPair pdf,
										const Connection& incident,
										const Spectrum& throughput,
										const float/* continuationPropability*/,
										const Spectrum& /*transmission*/,
										int /*numPhotons*/, float /*area*/) {
	float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
	bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
	this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
	this->throughput = throughput;
	if(prevVertex.is_hitable()) {
		// Compute as much as possible from the conversion factor.
		// At this point we do not know n and A for the photons. This quantities
		// are added in the kernel after the walk.
		float outCosAbs = ei::abs(prevVertex.get_geometric_factor(incident.dir));
		this->prevConversionFactor = orthoConnection ? outCosAbs : outCosAbs / incident.distanceSq;
	}
}

CUDA_FUNCTION void VcmVertexExt::update(const VcmPathVertex& thisVertex,
										const scene::Direction& /*excident*/,
										const VertexSample& sample,
										int numPhotons, float area) {
	// Sum up all previous relative probability (cached recursion).
	// Also see PBRT p.1015.
	prevRelativeProbabilitySum = prev_rel_sum(thisVertex, sample.pdf.back, numPhotons, area);
}

void CpuVcm::iterate() {
	auto scope = Profiler::core().start<CpuProfileState>("CPU VCM iteration", ProfileLevel::LOW);

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

void CpuVcm::post_reset() {
	ResetEvent resetFlags { { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
								ResetEvent::ALL : get_reset_event() } };
	init_rngs(m_outputBuffer.get_num_pixels());
	if(resetFlags.resolution_changed()) {
		m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_pathEndPoints.resize(m_outputBuffer.get_num_pixels());
	}
}

void CpuVcm::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	//u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	VcmPathVertex vertex;
	VcmPathVertex::create_light(&vertex, nullptr, p);
	const VcmPathVertex* previous = m_photonMap.insert(p.initVec, vertex);
	Spectrum throughput { 1.0f };
	float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

	int pathLen = 0;
	while(pathLen < m_params.maxPathLength-1) { // -1 because there is at least one segment on the view path
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, *previous, rnd, rndRoulette, true, throughput, vertex, sample,
				nullptr, numPhotons, mergeArea) != WalkResult::HIT)
			break;
		++pathLen;

		// Store a photon to the photon map
		previous = m_photonMap.insert(vertex.get_position(), vertex);
	}

	m_pathEndPoints[idx] = previous;
}

void CpuVcm::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeArea = ei::PI * mergeRadiusSq;
	u64 lightPathIdx = cn::WangHash{}(idx) % numPhotons;
	// Trace view path
	VcmPathVertex vertex[2];
	// Create a start for the path
	VcmPathVertex::create_camera(&vertex[0], nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	Spectrum throughput { 1.0f };
	int currentV = 0;
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
		const VcmPathVertex* lightVertex = m_pathEndPoints[lightPathIdx];
		while(lightVertex) {
			int pathLen = lightVertex->get_path_len() + 1 + viewPathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength) {
				Pixel outCoord = coord;
				auto conVal = connect(vertex[currentV], *lightVertex, m_sceneDesc, outCoord, numPhotons, mergeArea);
				if(outCoord != -1) {
					mAssert(!std::isnan(conVal.cosines) && !std::isnan(conVal.bxdfs.x) && !std::isnan(throughput.x) && !std::isnan(vertex[currentV].ext().throughput.x));
					m_outputBuffer.contribute<RadianceTarget>(outCoord, throughput * lightVertex->ext().throughput * conVal.cosines * conVal.bxdfs);
					m_outputBuffer.contribute<LightnessTarget>(outCoord, avg(throughput) * conVal.cosines);
				}
			}
			lightVertex = lightVertex->previous();
		}//*/

		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette, false, throughput, vertex[otherV],
				sample, nullptr, numPhotons, mergeArea) == WalkResult::CANCEL)
			break;
		++viewPathLen;
		currentV = otherV;

		// Evaluate direct hit of area ligths and the background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = vertex[currentV].get_emission(m_sceneDesc, vertex[1-currentV].get_position());
			if(emission.value != 0.0f) {
				float misWeight = get_mis_weight(vertex[currentV], emission.pdf, emission.emitPdf, numPhotons, ei::PI * mergeRadiusSq);
				emission.value *= misWeight;
			}
			mAssert(!std::isnan(emission.value.x));

			m_outputBuffer.contribute<RadianceTarget>(coord, throughput * emission.value);
			m_outputBuffer.contribute<LightnessTarget>(coord, avg(throughput) * ei::avg(emission.value));
		}//*/
		if(vertex[currentV].is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = vertex[currentV].get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			auto& photon = *photonIt;
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int pathLen = viewPathLen + photon.get_path_len();
			if(photon.get_path_len() > 0 && pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photon.get_position() - currentPos) < mergeRadiusSq) {
				// Radiance estimate
				Pixel tmpCoord;
				scene::Direction geoNormal = photon.get_geometric_normal();
				auto bsdf = vertex[currentV].evaluate(-photon.get_incident_direction(),
													  m_sceneDesc.media, tmpCoord, false,
													  &geoNormal);
				float misWeight = get_mis_weight(vertex[currentV], bsdf.pdf, photon, numPhotons, mergeArea);
				radiance += bsdf.value * photon.ext().throughput * misWeight;
			}
			++photonIt;
		}
		radiance /= mergeArea * numPhotons;
		m_outputBuffer.contribute<RadianceTarget>(coord, throughput * radiance);
		m_outputBuffer.contribute<LightnessTarget>(coord, avg(throughput) * ei::avg(radiance));
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuVcm::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer