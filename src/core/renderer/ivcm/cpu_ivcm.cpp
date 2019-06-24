﻿#include "cpu_ivcm.hpp"
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

#include <cn/rnd.hpp>

namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct IvcmVertexExt {
	AreaPdf incidentPdf;
	Spectrum throughput;
	AngularPdf pdfBack;
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor { 0.0f };


	CUDA_FUNCTION void init(const IvcmPathVertex& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosine,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput.weight;
		if(thisVertex.previous() && thisVertex.previous()->is_hitable()) {
			// Compute as much as possible from the conversion factor.
			// At this point we do not know n and A for the photons. This quantities
			// are added in the kernel after the walk.
			this->prevConversionFactor = float(
				thisVertex.previous()->convert_pdf(thisVertex.get_type(), AngularPdf{1.0f},
				{ incident, incidentDistance * incidentDistance }).pdf );
		}
	}

	CUDA_FUNCTION void update(const IvcmPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		pdfBack = pdf.back;
	}
};

// incidentF/incidentB: per vertex area pdfs
// n: path length in segments
// idx: index of merge vertex
float get_mis_weight_photon(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float mergeArea, int numPhotons) {
	if(idx == 0 || idx == n) return 0.0f;
	// Start with camera connection
	float relPdfSumV = 1.0f / (float(incidentF[1]) * mergeArea * numPhotons);
	// Collect merges and connections along view path
	for(int i = 1; i < idx; ++i) {
		float prevMerge = incidentB[i] / incidentF[i+1];
		float prevConnect = 1.0f / (float(incidentF[i+1]) * mergeArea * numPhotons);
		relPdfSumV = prevConnect + prevMerge * (1.0f + relPdfSumV);
	}
	// Collect merges/connect/hit along light path
	float relPdfSumL = 1.0f / (numPhotons * mergeArea * float(incidentB[n-1]));	// Connection
	relPdfSumL += incidentF[n] / incidentB[n] * relPdfSumL;						// Random hit
	for(int i = n-2; i >= idx; --i) {
		float prevMerge = incidentF[i+1] / incidentB[i];
		float prevConnect = 1.0f / (float(incidentB[i]) * mergeArea * numPhotons);
		relPdfSumL = prevConnect + prevMerge * (1.0f + relPdfSumL);
	}
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

// incidentF/incidentB: per vertex area pdfs
// n: path length in segments
// idx: index of first connection vertex
float get_mis_weight_connect(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float mergeArea, int numPhotons) {
	float relPdfSumV = 0.0f;
	// Collect merges and connections along view path
	for(int i = 1; i <= idx; ++i) {
		float prevConnect = incidentB[i] / incidentF[i];
		float curMerge = float(incidentB[i]) * mergeArea * numPhotons;
		relPdfSumV = curMerge + prevConnect * (1.0f + relPdfSumV);
	}
	// Collect merges/connect/hit along light path
	float relPdfSumL = incidentF[n] / incidentB[n];		// Random hit
	for( int i = n-1; i > idx; --i) {
		float prevConnect = incidentF[i] / incidentB[i];
		float curMerge = float(incidentF[i]) * mergeArea * numPhotons;
		relPdfSumL = curMerge + prevConnect * (1.0f + relPdfSumL);
	}
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

float get_mis_weight_rhit(const AreaPdf* incidentF, const AreaPdf* incidentB, int n,
	float mergeArea, int numPhotons) {
	// Collect all connects/merges along the view path only
	float relPdfSumV = 0.0f;
	for(int i = 1; i < n; ++i) {
		float prevConnect = incidentB[i] / incidentF[i];
		float curMerge = float(incidentB[i]) * mergeArea * numPhotons;
		relPdfSumV = curMerge + prevConnect * (1.0f + relPdfSumV);
	}
	float connectionRel = incidentB[n] / incidentF[n];
	relPdfSumV = connectionRel * (1.0f + relPdfSumV);
	return 1.0f / (1.0f + relPdfSumV);
}

// Fill a range in the incidentF/B arrays
void copy_path_values(AreaPdf* incidentF, AreaPdf* incidentB,
	const IvcmPathVertex* vert, Interaction prevType,
	AngularPdf pdfBack, ConnectionDir connectionDir,
	int begin, int end) {
	int step = begin < end ? 1 : -1;
	for(int i = begin; i != end; i += step) {
		incidentF[i] = vert->ext().incidentPdf;
		incidentB[i] = vert->convert_pdf(prevType, pdfBack, connectionDir).pdf;
		pdfBack = vert->ext().pdfBack;
		connectionDir = vert->get_incident_connection();
		vert = vert->previous();
	}
}


struct ConnectionValue { Spectrum bxdfs; float cosines; };
ConnectionValue connect(const IvcmPathVertex& path0, const IvcmPathVertex& path1,
						const scene::SceneDescriptor<Device::CPU>& scene,
						Pixel& coord, float mergeArea, int numPhotons,
						AreaPdf* incidentF, AreaPdf* incidentB
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!IvcmPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = IvcmPathVertex::get_connection(path0, path1);
	auto val0 = path0.evaluate( connection.dir, scene.media, coord, false);
	auto val1 = path1.evaluate(-connection.dir, scene.media, coord, true);
	// Cancel reprojections outside the screen
	if(coord.x == -1) return {Spectrum{0.0f}, 0.0f};
	Spectrum bxdfProd = val0.value * val1.value;
	float cosProd = val0.cosOut * val1.cosOut;//TODO: abs?
	mAssert(cosProd >= 0.0f);
	mAssert(!isnan(bxdfProd.x));
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
		// Shadow test
		if(!scene::accel_struct::any_intersection(
				scene,connection.v0, path1.get_position(connection.v0),
				path0.get_geometric_normal(),  path1.get_geometric_normal(), connection.dir)) {
			bxdfProd *= scene.media[path0.get_medium(connection.dir)].get_transmission(connection.distance);
			// Collect quantities for MIS
			int pl0 = path0.get_path_len();
			incidentB[pl0] = path0.convert_pdf(path1.get_type(), val1.pdf.forw, connection).pdf;
			incidentF[pl0] = path0.ext().incidentPdf;
			copy_path_values(incidentF, incidentB, path0.previous(), path0.get_type(),
				val0.pdf.back, path0.get_incident_connection(), pl0-1, -1);
			int pathLen = pl0 + path1.get_path_len() + 1;
			incidentF[pl0+1] = path1.convert_pdf(path0.get_type(), val0.pdf.forw, connection).pdf;
			incidentB[pl0+1] = path1.ext().incidentPdf;
			copy_path_values(incidentB, incidentF, path1.previous(), path1.get_type(),
				val1.pdf.back, path1.get_incident_connection(), pl0 + 2, pathLen + 1);
			float misWeight = get_mis_weight_connect(incidentF, incidentB, pathLen, pl0, mergeArea, numPhotons);
			return {bxdfProd * (misWeight / connection.distanceSq), cosProd};
		}
	}
	return {Spectrum{0.0f}, 0.0f};
}

Spectrum merge(const IvcmPathVertex& viewPath, const IvcmPathVertex& photon,
			   const scene::SceneDescriptor<Device::CPU>& scene,
			   float mergeArea, int numPhotons,
			   AreaPdf* incidentF, AreaPdf* incidentB) {
	// Radiance estimate
	Pixel tmpCoord;
	scene::Direction geoNormal = photon.get_geometric_normal();
	auto bsdf = viewPath.evaluate(-photon.get_incident_direction(),
								   scene.media, tmpCoord, false,
								   &geoNormal);
	// Collect values for mis
	int pl0 = viewPath.get_path_len();
	int pathLen = pl0 + photon.get_path_len();
	incidentF[pl0] = viewPath.ext().incidentPdf;
	incidentB[pl0] = photon.ext().incidentPdf;
	copy_path_values(incidentF, incidentB, viewPath.previous(), viewPath.get_type(),
		bsdf.pdf.back, viewPath.get_incident_connection(), pl0 - 1, -1);
	copy_path_values(incidentB, incidentF, photon.previous(), photon.get_type(),
		bsdf.pdf.forw, photon.get_incident_connection(), pl0 + 1, pathLen + 1);
	float misWeight = get_mis_weight_photon(incidentF, incidentB, pathLen, pl0, mergeArea, numPhotons);
	return bsdf.value * photon.ext().throughput * misWeight;
}

} // namespace ::


CpuIvcm::CpuIvcm() {}
CpuIvcm::~CpuIvcm() {}

void CpuIvcm::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU IVCM iteration", ProfileLevel::LOW);

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
		AreaPdf* incidentF = m_tmpPathProbabilities.data() + get_current_thread_idx() * 2 * (m_params.maxPathLength + 1);
		AreaPdf* incidentB = incidentF + (m_params.maxPathLength + 1);
		IvcmPathVertex* vertexBuffer = m_tmpViewPathVertices.data() + get_current_thread_idx() * (m_params.maxPathLength + 1);
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
					 pixel, numPhotons, currentMergeRadius, incidentF, incidentB, vertexBuffer);
	}
}

void CpuIvcm::post_reset() {
	ResetEvent resetFlags { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
								ResetEvent::ALL : get_reset_event() };
	init_rngs(m_outputBuffer.get_num_pixels());
	if(resetFlags.resolution_changed()) {
		m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_pathEndPoints.resize(m_outputBuffer.get_num_pixels());
	}
	if(resetFlags.is_set(ResetEvent::PARAMETER)) {
		m_tmpPathProbabilities.resize(get_thread_num() * 2 * (m_params.maxPathLength + 1));
		m_tmpViewPathVertices.resize(get_thread_num() * (m_params.maxPathLength + 1));
	}
}

void CpuIvcm::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2_1 rndStart { m_rngs[idx].next(), m_rngs[idx].next() };
	//u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	IvcmPathVertex vertex;
	IvcmPathVertex::create_light(&vertex, nullptr, p);
	const IvcmPathVertex* previous = m_photonMap.insert(p.pos.position, vertex);
	math::Throughput throughput;
	float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

	int pathLen = 0;
	while(pathLen < m_params.maxPathLength-1) { // -1 because there is at least one segment on the view path
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, *previous, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
			break;
		++pathLen;

		// Store a photon to the photon map
		previous = m_photonMap.insert(vertex.get_position(), vertex);
	}

	m_pathEndPoints[idx] = previous;
}

void CpuIvcm::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius,
					 AreaPdf* incidentF, AreaPdf* incidentB, IvcmPathVertex* vertexBuffer) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeArea = ei::PI * mergeRadiusSq;
	u64 lightPathIdx = cn::WangHash{}(idx) % numPhotons;
	// Trace view path
	// Create a start for the path
	IvcmPathVertex* currentVertex = vertexBuffer;
	IvcmPathVertex::create_camera(currentVertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	math::Throughput throughput;
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
		const IvcmPathVertex* lightVertex = m_pathEndPoints[lightPathIdx];
		while(lightVertex) {
			int pathLen = lightVertex->get_path_len() + 1 + viewPathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength) {
				Pixel outCoord = coord;
				auto conVal = connect(*currentVertex, *lightVertex, m_sceneDesc, outCoord, mergeArea, numPhotons, incidentF, incidentB);
				mAssert(!isnan(conVal.cosines) && !isnan(conVal.bxdfs.x) && !isnan(throughput.weight.x) && !isnan(currentVertex->ext().throughput.x));
				m_outputBuffer.contribute(outCoord, throughput, math::Throughput{lightVertex->ext().throughput, 1.0f}, conVal.cosines, conVal.bxdfs);
			}
			lightVertex = lightVertex->previous();
		}//*/

		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, *currentVertex, rnd, rndRoulette, false, throughput,
				*(currentVertex + 1), sample) == WalkResult::CANCEL)
			break;
		++viewPathLen;
		++currentVertex;

		// Evaluate direct hit of area ligths and the background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = currentVertex->get_emission(m_sceneDesc, currentVertex->previous()->get_position());
			if(emission.value != 0.0f) {
				incidentF[viewPathLen] = currentVertex->ext().incidentPdf;
				incidentB[viewPathLen] = emission.emitPdf;
				copy_path_values(incidentF, incidentB, currentVertex->previous(), currentVertex->get_type(),
					emission.pdf, currentVertex->get_incident_connection(), viewPathLen - 1, -1);
				float misWeight = get_mis_weight_rhit(incidentF, incidentB, viewPathLen, mergeArea, numPhotons);
				emission.value *= misWeight;
			}
			mAssert(!isnan(emission.value.x));
			m_outputBuffer.contribute(coord, throughput, emission.value, currentVertex->get_position(),
									currentVertex->get_normal(), currentVertex->get_albedo());
		}//*/
		if(currentVertex->is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = currentVertex->get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			auto& photon = *photonIt;
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int pathLen = viewPathLen + photon.get_path_len();
			if(photon.get_path_len() > 0 && pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photon.get_position() - currentPos) < mergeRadiusSq) {
				radiance += merge(*currentVertex, photon, m_sceneDesc, mergeArea, numPhotons,
								  incidentF, incidentB);
			}
			++photonIt;
		}
		radiance /= mergeArea * numPhotons;
		m_outputBuffer.contribute(coord, throughput, radiance, scene::Point{0.0f},
			scene::Direction{0.0f}, Spectrum{0.0f});//*/
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuIvcm::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer