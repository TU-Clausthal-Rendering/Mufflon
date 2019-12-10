#include "cpu_neb.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/parameter.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

namespace mufflon::renderer {

namespace {
float get_photon_path_chance(const AngularPdf pdf) {
	return 1.0f;
	return float(pdf) / (10.0f + float(pdf));
}

#ifdef NEB_KDTREE
float get_density(const scene::accel_struct::KdTree<char, 3>& tree, const ei::Vec3& currentPos, const ei::Vec3&) {
	int idx[4];
	float distSq[4];
	tree.query_euclidean(currentPos, 4, idx, distSq);
	float queryArea = (distSq[3]) * ei::PI / 3.0f;
	//float queryArea = (distSq[3] - distSq[2]) * ei::PI / 1.0f;
	return 1.0f / queryArea;
}
#else
float get_density(const std::unique_ptr<data_structs::DmOctree<>>& octree, const ei::Vec3& currentPos, const ei::Vec3& normal) {
	float density = octree->get_density(currentPos, normal);
//	density = ei::max(1.0f / (numStdPhotonPaths * (m_currentIteration+1)),
//		m_density->get_density_interpolated(vertex.get_position(), vertex.get_geometric_normal()));
	mAssert(density < 1e38f && density > 0.0f);
	return density;
}
#endif

} // namespace ::


struct NebVertexExt {
	Spectrum throughput;
	AngularPdf pdfBack;
	AreaPdf incidentPdf { 0.0f };
	union {
		int pixelIndex;
		float rnd;		// An additional input random value (first vertex of light paths only).
	};
	float density { -1.0f };

	inline CUDA_FUNCTION void init(const PathVertex<NebVertexExt>& /*thisVertex*/,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		this->throughput = Spectrum{1.0f};
	}

	inline CUDA_FUNCTION void update(const PathVertex<NebVertexExt>& prevVertex,
							  const PathVertex<NebVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& throughput,
							  const float /*continuationPropability*/,
							  const Spectrum& /*transmission*/) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
		this->throughput = throughput;
	}

	void update(const PathVertex<NebVertexExt>& /*thisVertex*/,
				const scene::Direction& /*excident*/,
				const VertexSample& sample) {
		pdfBack = sample.pdf.back;
	}
};


class NebPathVertex : public PathVertex<NebVertexExt> {
public:
	// Overload the vertex sample operator to have more RR control.
	VertexSample sample(const ei::Box& sceneBounds,
						const scene::materials::Medium* media,
						const math::RndSet2_1& rndSet,
						bool adjoint) const {
		VertexSample s = PathVertex<NebVertexExt>::sample(sceneBounds, media, rndSet, adjoint);
		// Conditionally kill the photon path tracing if we are on the
		// NEE vertex. Goal: trace only photons which we cannot NEE.
		if(adjoint && get_path_len() == 1) {
			float keepChance = get_photon_path_chance(s.pdf.forw);
			if(keepChance < ext().rnd) {
				s.type = math::PathEventType::INVALID;
			} else {
				s.throughput /= keepChance;
			}
		}
		return s;
	}
};

namespace {

int get_photon_split_count(const NebPathVertex& /*vertex*/, float maxFlux, float targetFlux) {
	return 1; // Disabled
	float smoothness = 1.0f;//ei::min(1e30f, vertex.get_pdf_max() * ei::PI);
	if(smoothness < 1e-3f) // Max-pdf cannot be that small (except the surface does not reflect at all)
		return 0;
	return ei::ceil(maxFlux / (smoothness * targetFlux));
}

struct PhotonConversionFactors { float toFlux; int photonCount; };
PhotonConversionFactors
get_photon_conversion_factors(const NebPathVertex& vertex,
							  const CpuNextEventBacktracking::NeeDesc& nee, const float targetFlux) {
	float toFlux = ei::abs(vertex.get_geometric_factor(nee.direction)) / vertex.ext().density;
	float maxFlux = max(nee.irradiance) * toFlux;
	int photonCount = get_photon_split_count(vertex, maxFlux, targetFlux);
	return { toFlux / ei::max(1,photonCount), photonCount };
}

float mis_stdphoton(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float density, float mergeArea, int numPhotons) {
	if(idx == 0 || idx == n) return 0.0f;
	float relPdfSumV = 0.0f;
	float toNebMerge = density / (numPhotons * float(incidentB[n-1]));
	// Collect merges along view path
	for(int i = 1; i < idx; ++i) {
		float prevMerge = incidentB[i] / incidentF[i+1];
		relPdfSumV = prevMerge * (1.0f + toNebMerge + relPdfSumV);
	}
	// Collect merges/nee/random hit along light path
	float relPdfSumL = 1.0f / (numPhotons * mergeArea * float(incidentB[n-1]));	// Connection
	relPdfSumL += incidentF[n] / incidentB[n] * relPdfSumL;						// Random hit
	for(int i = n-2; i >= idx; --i) {
		float prevMerge = incidentF[i+1] / incidentB[i];
		relPdfSumL = prevMerge * (1.0f + relPdfSumL);
		relPdfSumL += toNebMerge;		// Other merge at the current point possible
	}
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

float mis_nebphoton(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float density, float mergeArea, int numPhotons, bool stdPhotons) {
	if(idx == 0 || idx == n) return 0.0f;
	float relPdfSumV = 0.0f;
	float toStdMerge = stdPhotons ? (numPhotons * float(incidentB[n-1])) / density : 0.0f;
	// Collect merges along view path
	for(int i = 1; i < idx; ++i) {
		float prevMerge = incidentB[i] / incidentF[i+1];
		relPdfSumV = prevMerge * (1.0f + toStdMerge + relPdfSumV);
	}
	// Collect merges/nee/random hit along light path
	float relPdfSumL = 1.0f / (mergeArea * density);			// Connection
	relPdfSumL += incidentF[n] / incidentB[n] * relPdfSumL;		// Random hit
	relPdfSumL += toStdMerge;
	for(int i = n-2; i >= idx; --i) {
		float prevMerge = incidentF[i+1] / incidentB[i];
		relPdfSumL = prevMerge * relPdfSumL;
		relPdfSumL += 1.0f + toStdMerge;
	}
	return 1.0f / (relPdfSumV + relPdfSumL);
}

float mis_nee(const AreaPdf* incidentF, const AreaPdf* incidentB, int n,
	float density, float mergeArea, int numPhotons, bool stdPhotons) {
	float relPdfSumL = incidentF[n] / incidentB[n];		// Random hit
	// Collect merges along view path
	float toStdMerge = stdPhotons ? numPhotons * mergeArea * float(incidentB[n-1]) : 0.0f;
	float toNebMerge = mergeArea * density;
	float relPdfSumV = 0.0f;
	for(int i = 1; i < n-1; ++i) {
		float prevMerge = incidentB[i] / incidentF[i+1];
		relPdfSumV = prevMerge * (toNebMerge + toStdMerge + relPdfSumV);
	}
	relPdfSumV += toStdMerge;
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

float mis_rhit(const AreaPdf* incidentF, const AreaPdf* incidentB, int n,
	float density, float mergeArea, int numPhotons, bool stdPhotons) {
	float relConnection = incidentB[n] / incidentF[n];
	return mis_nee(incidentF, incidentB, n, density,
		mergeArea, numPhotons, stdPhotons) / relConnection;
}
} // namespace ::


void CpuNextEventBacktracking::sample_view_path(const Pixel coord, const int pixelIdx) {
	Spectrum throughput{ 1.0f };
	NebPathVertex* previous = nullptr;
	NebPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	NebPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[pixelIdx].next());

	int pathLen = 0;
	do {
		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { m_rngs[pixelIdx].next(), m_rngs[pixelIdx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixelIdx].next()));
		NebPathVertex& sourceVertex = previous ? *previous : vertex;	// Make sure the update function is called for the correct vertex.
		if(walk(m_sceneDesc, sourceVertex, rnd, rndRoulette, false, throughput, vertex, sample) == WalkResult::CANCEL)
			break;
		++pathLen;
		if(pathLen >= m_params.minPathLength) {
			EmissionValue emission = vertex.get_emission(m_sceneDesc, lastPosition);
			// In the case that there is no previous, there is also no MIS and we
			// contribute directly.
			if(!previous) {
				m_outputBuffer.contribute<RadianceTarget>(coord, throughput * emission.value);
			} else if(emission.value != 0.0f) {
				EmissionDesc emDesc;
				emDesc.previous = static_cast<const NebPathVertex*>(vertex.previous());
				emDesc.radiance = throughput * emission.value;
				emDesc.incidentPdf = vertex.ext().incidentPdf;
				emDesc.startPdf = pathLen == 1 ? AreaPdf{0.0f} : emission.connectPdf;
				// Special case: when photons are traced they use the emit pdf and not the connect pdf.
				// Fortunatelly, the emDesc.samplePdf is only used for the tracing events, so we can
				// include the difference between the two events in this PDF.
				emDesc.samplePdf = pathLen == 1 ? AngularPdf{0.0f} : emission.pdf * (emission.emitPdf / emission.connectPdf);
				emDesc.incident = sample.excident;
				emDesc.incidentDistSq = vertex.get_incident_dist_sq();
				emDesc.maxIrradiance = max(emission.value)
					* ei::abs(vertex.previous()->get_geometric_factor(sample.excident))
					/ (float(emission.connectPdf) * emDesc.incidentDistSq);
				u32 idx = m_selfEmissionCount.fetch_add(1);
				m_selfEmissiveEndVertices[idx] = emDesc;
			}
		}
		if(vertex.is_end_point()) break;

		vertex.ext().pixelIndex = pixelIdx;
		// Using NEE on random hit vertices has low contribution in general.
		// Therefore, we store the preevaluated emission only.
		if(pathLen < m_params.maxPathLength) {
			previous = m_viewVertexMap.insert(vertex.get_position(), vertex);
			if(previous == nullptr) { mAssert(false); break; }	// OVERFLOW
#ifdef NEB_KDTREE
			m_density.insert(vertex.get_position(), 0);
#else
			m_density->increase_count(vertex.get_position(), vertex.get_geometric_normal());
#endif
		}
	} while(pathLen < m_params.maxPathLength);
}

// Create the backtracking path
void CpuNextEventBacktracking::sample_photon_path(float photonMergeArea, math::Rng& rng,
	const NebPathVertex& vertex, const NeeDesc& nee,
	AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons) {
	//const float mergeRadiusSq = photonMergeArea / ei::PI;
	// Precomupte the irradiance -> flux factor and
	// pplit photons on rough surfaces if their density is much smaller than necessary
	auto pFactors = get_photon_conversion_factors(vertex, nee, m_targetFlux);
	Spectrum flux = nee.irradiance * pFactors.toFlux;
	// Clamping for bad photons (due to errors in the density estimation)
	//float expectedFluxMax = max(flux);
	//flux *= ei::min(expectedFluxMax, m_targetFlux) / expectedFluxMax;
	// Fill the pdf arrays on the fly.
	int pdfHead = m_params.maxPathLength;
	incidentB[pdfHead] = nee.creationPdf;
	incidentB[pdfHead-1] = nee.backPdf;
	const float sourceDensity = vertex.ext().density * pFactors.photonCount;
	float prevConversionFactor = nee.conversion;
	for(int i = 0; i < pFactors.photonCount; ++i) {
		pdfHead = m_params.maxPathLength - 1;
		// Prepare a start vertex to begin the sampling of the photon event.
		NebPathVertex virtualLight = vertex;
		virtualLight.set_incident_direction(-nee.direction, nee.distance);

		// Trace a path
		virtualLight.ext().rnd = math::sample_uniform(u32(rng.next()));
		int lightPathLength = 1;
		Spectrum lightThroughput { 1.0f };
		scene::Direction prevNormal = virtualLight.get_geometric_normal();
		while(lightPathLength < m_params.maxPathLength-1) {
			math::RndSet2_1 rnd { rng.next(), rng.next() };
			float rndRoulette = math::sample_uniform(u32(rng.next()));
			VertexSample sample;
			if(walk(m_sceneDesc, virtualLight, rnd, rndRoulette, true, lightThroughput, virtualLight, sample) != WalkResult::HIT)
				break;
			++lightPathLength;
			virtualLight.set_path_len(lightPathLength);
			
			incidentF[pdfHead+1] = AreaPdf{float(sample.pdf.back) * prevConversionFactor};
			--pdfHead;
			prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / virtualLight.get_incident_dist_sq();
			prevNormal = virtualLight.get_geometric_normal();
			merge_importons(photonMergeArea, virtualLight, incidentF, incidentB,
				numPhotons, false, lightThroughput * flux, pdfHead,
				prevConversionFactor, sourceDensity);
		}
	}
}

void CpuNextEventBacktracking::sample_std_photon(int idx, int numPhotons, u64 seed, float photonMergeArea,
	AreaPdf* incidentF, AreaPdf* incidentB) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	//float mergeRadiusSq = photonMergeArea / ei::PI;
	//u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	NebPathVertex vertex[2];
	NebPathVertex::create_light(&vertex[0], nullptr, p);

	Spectrum throughput { 1.0f };
	scene::Direction prevNormal = vertex[0].get_geometric_normal();
	float prevConversionFactor = 0.0f;
	float sourceDensity = 0.0f;
	int lightPathLength = 0;
	int currentV = 0;
	// Fill the pdf arrays on the fly.
	int pdfHead = m_params.maxPathLength;
	incidentB[pdfHead] = vertex[0].ext().incidentPdf;
	while(lightPathLength < m_params.maxPathLength-1) { // -1 because there is at least one segment on the view path
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		math::RndSet2 rndRoulette { m_rngs[idx].next() };
		vertex[currentV].ext().rnd = rndRoulette.u1;
		VertexSample sample;
		if(walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette.u0, true, throughput, vertex[1-currentV], sample) != WalkResult::HIT)
			break;
		++lightPathLength;
		currentV = 1-currentV;

		if(lightPathLength == 1) {
			sourceDensity = get_density(m_density, vertex[currentV].get_position(), vertex[currentV].get_geometric_normal());
			// Need to simulate a next event irradiance for the first segment to find the split factor.
			// maxFlux = max(nee.irradiance) * |vertex.get_geometric_factor(nee.direction)| / density;
			const float inCos = ei::abs(vertex[currentV].get_geometric_factor(sample.excident));
			Pixel tmpCoord;
			auto evalVal = vertex[0].evaluate(sample.excident, m_sceneDesc.media, tmpCoord);
			const float maxIrradiance = max(evalVal.value) * evalVal.cosOut / vertex[currentV].get_incident_dist_sq();
			const float maxFlux = maxIrradiance * inCos / sourceDensity;
			sourceDensity *= get_photon_split_count(vertex[currentV], maxFlux, m_targetFlux);
		}
		if(pdfHead < m_params.maxPathLength)
			incidentF[pdfHead+1] = AreaPdf{float(sample.pdf.back) * prevConversionFactor};
		--pdfHead;
		prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / vertex[currentV].get_incident_dist_sq();
		prevNormal = vertex[currentV].get_geometric_normal();
		merge_importons(photonMergeArea, vertex[currentV], incidentF, incidentB, numPhotons,
			true, throughput / numPhotons, pdfHead, prevConversionFactor, sourceDensity);
	}
}

void CpuNextEventBacktracking::merge_importons(float photonMergeArea, const NebPathVertex& lvertex,
	AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons, bool isStd,
	const Spectrum& flux, int pdfHead, float prevConversionFactor, float sourceDensity) {

	float mergeRadiusSq = photonMergeArea / ei::PI;
	incidentB[pdfHead] = lvertex.ext().incidentPdf;

	// Merge with view path vertices
	auto vertexIt = m_viewVertexMap.find_first(lvertex.get_position());
	scene::Direction lightNormal = lvertex.get_geometric_normal();
	int lightPathLength = lvertex.get_path_len();
	while(vertexIt) {
		auto& vvertex = *vertexIt;
		int viewPathLen = vvertex.get_path_len();
		int pathLen = lightPathLength + viewPathLen;
		if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
			&& lensq(vvertex.get_position() - lvertex.get_position()) < mergeRadiusSq) {
			Pixel tmpCoord;
			auto bsdf = vvertex.evaluate(-lvertex.get_incident_direction(),
										  m_sceneDesc.media, tmpCoord, false,
										  &lightNormal);
			if(any(greater(bsdf.value, 0.0f))) {
				incidentF[pdfHead+1] = AreaPdf{float(bsdf.pdf.forw) * prevConversionFactor};
				incidentF[pdfHead] = vvertex.ext().incidentPdf;
				int pdfOffset = pdfHead - viewPathLen;
				AngularPdf pdfBack = bsdf.pdf.back;
				ConnectionDir connection = vvertex.get_incident_connection();
				const auto* vert = vvertex.previous();
				for(int i = viewPathLen - 1; i > 0; --i) {
					incidentF[i+pdfOffset] = vert->ext().incidentPdf;
					incidentB[i+pdfOffset] = vert->convert_pdf(Interaction::SURFACE, pdfBack, connection).pdf;
					pdfBack = vert->ext().pdfBack;
					connection = vert->get_incident_connection();
					vert = vert->previous();
				}
				incidentF[pdfOffset] = AreaPdf{1.0f};
				incidentB[pdfOffset] = AreaPdf{0.0f};
				float misWeight;
				if(isStd)
					misWeight = mis_stdphoton(incidentF+pdfOffset, incidentB+pdfOffset, pathLen, viewPathLen,
						sourceDensity, photonMergeArea, numPhotons);
				else
					misWeight = mis_nebphoton(incidentF+pdfOffset, incidentB+pdfOffset, pathLen, viewPathLen,
						sourceDensity, photonMergeArea, numPhotons, m_params.stdPhotons);
				Spectrum radiance = vvertex.ext().throughput * bsdf.value * flux * (misWeight  / photonMergeArea);
				Pixel coord { vvertex.ext().pixelIndex % m_outputBuffer.get_width(),
							  vvertex.ext().pixelIndex / m_outputBuffer.get_width() };
				m_outputBuffer.contribute<RadianceTarget>(coord, radiance);
			} // BSDF > 0
		} // Pathlen and photon distance
		++vertexIt;
	}
}

CpuNextEventBacktracking::NeeDesc CpuNextEventBacktracking::compute_nee(math::Rng& rng, const NebPathVertex& vertex) {
	u64 neeSeed = rng.next();
	math::RndSet2 neeRnd = rng.next();
	auto nee = scene::lights::connect(m_sceneDesc, 0, 1, neeSeed, vertex.get_position(), neeRnd);
	if(nee.cosOut != 0) nee.diffIrradiance *= nee.cosOut;
	bool anyhit = scene::accel_struct::any_intersection(
						m_sceneDesc, vertex.get_position(), nee.position,
						vertex.get_geometric_normal(), nee.geoNormal, nee.dir.direction);
	// Make sure the vertex for which we did the NEE knows it is shadowed.
	if(anyhit)
		nee.diffIrradiance = Spectrum{0.0f};
	return {
		nee.diffIrradiance,
		nee.dir.direction,
		nee.cosOut / nee.distSq,
		nee.creationPdf,
		nee.dir.pdf.to_area_pdf(ei::abs(vertex.get_geometric_factor(nee.dir.direction)), nee.distSq),
		nee.dist
	};
}

void CpuNextEventBacktracking::evaluate_nee(const NebPathVertex& vertex,
											const NeeDesc& nee,
											AreaPdf* incidentF, AreaPdf* incidentB,
											int numPhotons, float photonMergeArea) {
	int n = vertex.get_path_len() + 1;
	// If path length is already too large there will be no contribution from this vertex.
	// It only exists for the sake of random hit evaluation (and as additional sample).
	if(n < m_params.minPathLength || n > m_params.maxPathLength)
		return;
	Pixel tmpCoord;
	auto bsdf = vertex.evaluate(nee.direction, m_sceneDesc.media, 
								tmpCoord, false, nullptr);
	// MIS compares against all previous merges (there are no feature ones)
	incidentF[n] = AreaPdf{ float(bsdf.pdf.forw) * nee.conversion };
	incidentB[n] = nee.creationPdf;
	incidentF[n-1] = vertex.ext().incidentPdf;
	incidentB[n-1] = nee.backPdf;
	const auto* vert = vertex.previous();
	AngularPdf pdfBack = bsdf.pdf.back;
	ConnectionDir connection = vertex.get_incident_connection();
	for(int i = n-2; i > 0; --i) {
		incidentF[i] = vert->ext().incidentPdf;
		incidentB[i] = vert->convert_pdf(Interaction::SURFACE, pdfBack, connection).pdf;
		pdfBack = vert->ext().pdfBack;
		connection = vert->get_incident_connection();
		vert = vert->previous();
	}
	incidentF[0] = AreaPdf{1.0f};
	incidentB[0] = AreaPdf{0.0f};
	const float sourceDensity = vertex.ext().density * get_photon_conversion_factors(vertex, nee, m_targetFlux).photonCount;
	float misWeight = mis_nee(incidentF, incidentB, n, sourceDensity, photonMergeArea, numPhotons, m_params.stdPhotons);
	Spectrum radiance = vertex.ext().throughput * (misWeight * bsdf.cosOut) * bsdf.value * nee.irradiance;
	Pixel coord { vertex.ext().pixelIndex % m_outputBuffer.get_width(),
				  vertex.ext().pixelIndex / m_outputBuffer.get_width() };
	m_outputBuffer.contribute<RadianceTarget>(coord, radiance);

}

Spectrum CpuNextEventBacktracking::finalize_emission(float photonMergeArea, const EmissionDesc& emission,
	AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons) {
	if(emission.previous) {
		int n = emission.previous->get_path_len()+1;
		incidentF[n] = emission.incidentPdf;
		incidentB[n] = emission.startPdf;
		const PathVertex<NebVertexExt>* vert = emission.previous;
		AngularPdf pdfBack = emission.samplePdf;
		ConnectionDir connection { emission.incident, emission.incidentDistSq };
		for(int i = n-1; i > 0; --i) {
			Interaction itype = connection.distanceSq == 0.0f ? Interaction::LIGHT_ENVMAP : Interaction::SURFACE;
			incidentF[i] = vert->ext().incidentPdf;
			incidentB[i] = vert->convert_pdf(itype, pdfBack, connection).pdf;
			pdfBack = vert->ext().pdfBack;
			connection = vert->get_incident_connection();
			vert = vert->previous();
		}
		incidentF[0] = AreaPdf{1.0f};
		incidentB[0] = AreaPdf{0.0f};
		const float maxFlux = emission.maxIrradiance / emission.previous->ext().density;
		const float sourceDensity = emission.previous->ext().density * get_photon_split_count(*emission.previous, maxFlux, m_targetFlux);
		float misWeight = mis_rhit(incidentF, incidentB, n,
			sourceDensity, photonMergeArea, numPhotons, m_params.stdPhotons);
		return emission.radiance * misWeight;
	} else
		return emission.radiance;
}

CpuNextEventBacktracking::CpuNextEventBacktracking() {
}

static int pairing(int a, int b) {
	return b + (a + b) * (a + b + 1) / 2;
}

void CpuNextEventBacktracking::iterate() {
	auto scope = Profiler::core().start<CpuProfileState>("CPU NEB iteration", ProfileLevel::HIGH);

	m_sceneDesc.lightTree.posGuide = m_params.neeUsePositionGuide;

	float photonMergeRadiusSq = ei::sq(m_params.mergeRadius * m_sceneDesc.diagSize);
	float photonMergeArea = photonMergeRadiusSq * ei::PI;
	m_viewVertexMap.clear(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
	m_selfEmissionCount.store(0);
#ifndef NEB_KDTREE
	m_density->set_iteration(m_currentIteration + 1);
#else
	m_density.clear();
#endif

	u64 photonSeed = m_rngs[0].next();

	std::vector<AreaPdf> tmpPdfMem((m_params.maxPathLength+1)*2*get_thread_num());

	// First pass: distribute and store view path vertices.
	// For each vertex compute the next event estimate, but do not contribute yet.
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		Pixel coord { pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		sample_view_path(coord, pixel);
	}
	i32 numViewVertices = m_viewVertexMap.size();

#ifdef NEB_KDTREE
	m_density.build();
#else
	// In the first iteration, the octree has a bad quality, because on each split
	// the distribution information is lost.
	// In this case we use the current allocation and readd all photons
	if(m_currentIteration == 0) {
		m_density->clear_counters();
#pragma PARALLEL_FOR
		for(i32 i = 0; i < numViewVertices; ++i)
			m_density->increase_count(m_viewVertexMap.get_data_by_index(i).get_position(), m_viewVertexMap.get_data_by_index(i).get_geometric_normal());
	}
#endif

	// Second pass: query densities for all vertices
	int numStdPhotonPaths = m_outputBuffer.get_num_pixels();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		vertex.ext().density = get_density(m_density, vertex.get_position(), vertex.get_geometric_normal());
		if(vertex.get_path_len() == 1) {
			Pixel coord { vertex.ext().pixelIndex % m_outputBuffer.get_width(),
						  vertex.ext().pixelIndex / m_outputBuffer.get_width() };
			m_outputBuffer.contribute<DensityTarget>(coord, vertex.ext().density);
		}
	}

	// Third pass: merge NEEs and backtrack. For each stored vertex find all other
	// vertices in the neighborhood and average the NEE result. Then trace a light path
	// beginning with a virtual source using the NEE direction as incident direction.
	// Store the new vertices in a standard photon map.
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		AreaPdf* incidentPdfsF = tmpPdfMem.data() + (m_params.maxPathLength+1)*2*get_current_thread_idx();
		AreaPdf* incidentPdfsB = incidentPdfsF + m_params.maxPathLength+1;
		int rngIndex = pairing(vertex.ext().pixelIndex, vertex.get_path_len());
		rngIndex = ei::mod(rngIndex, m_outputBuffer.get_num_pixels());
		//int rngIndex = i % m_outputBuffer.get_num_pixels();

		// Compute NEE and contribute
		NeeDesc nee = compute_nee(m_rngs[rngIndex], vertex);
		// Skip shadowed events
		if(any(greater(nee.irradiance, 0.0f))) {
			evaluate_nee(vertex, nee, incidentPdfsF, incidentPdfsB, numStdPhotonPaths, photonMergeArea);
			sample_photon_path(photonMergeArea, m_rngs[rngIndex], vertex, nee, incidentPdfsF, incidentPdfsB, numStdPhotonPaths);
		}
	}//*/

	// Additional standard photons
	if(m_params.stdPhotons) {
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < numStdPhotonPaths; ++pixel) {
			AreaPdf* incidentPdfsF = tmpPdfMem.data() + (m_params.maxPathLength+1)*2*get_current_thread_idx();
			AreaPdf* incidentPdfsB = incidentPdfsF + m_params.maxPathLength+1;
			sample_std_photon(pixel, numStdPhotonPaths, photonSeed, photonMergeArea, incidentPdfsF, incidentPdfsB);
		}
	}//*/

	// Finialize the evaluation of emissive end vertices.
	// It is necessary to do this after the density estimate for a correct mis.
	i32 selfEmissionCount = m_selfEmissionCount.load();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < selfEmissionCount; ++i) {
		AreaPdf* incidentPdfsF = tmpPdfMem.data() + (m_params.maxPathLength+1)*2*get_current_thread_idx();
		AreaPdf* incidentPdfsB = incidentPdfsF + m_params.maxPathLength+1;
		Spectrum emission = finalize_emission(photonMergeArea, m_selfEmissiveEndVertices[i], incidentPdfsF, incidentPdfsB, numStdPhotonPaths);
		Pixel coord { m_selfEmissiveEndVertices[i].previous->ext().pixelIndex % m_outputBuffer.get_width(),
					  m_selfEmissiveEndVertices[i].previous->ext().pixelIndex / m_outputBuffer.get_width() };
		m_outputBuffer.contribute<RadianceTarget>(coord, emission);
	}//*/

	logPedantic("[NEB] Memory occupation    View-Vertices: ", m_viewVertexMap.size() * 100.0f / float(m_viewVertexMap.capacity()),
				"% | Octree: ", m_density->size() * 100.0f / float(m_density->capacity()), "%.");
#ifdef NEB_KDTREE
	logPedantic("[NEB] KD-Tree depth: ", m_density.compute_depth(), " optimal: ", ei::ilog2(m_density.size())+1);
#endif
}

void CpuNextEventBacktracking::post_reset() {
	ResetEvent resetFlags { { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
							ResetEvent::ALL : get_reset_event() } };
	init_rngs(m_outputBuffer.get_num_pixels());
	//int countHeuristic = m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1) * 2; // Save count
	int countHeuristic = m_outputBuffer.get_num_pixels() * ei::ceil(logf(float(m_params.maxPathLength)) * 4.0f);
	m_viewVertexMapManager.resize(countHeuristic);
	m_viewVertexMap = m_viewVertexMapManager.acquire<Device::CPU>();
	// There is at most one emissive end vertex per path
	m_selfEmissiveEndVertices.resize(m_outputBuffer.get_num_pixels());
#ifdef NEB_KDTREE
	m_density.reserve(countHeuristic);
#else
	if(resetFlags.geometry_changed())
		m_density = std::make_unique<data_structs::DmOctree<>>(m_sceneDesc.aabb,
			1024 * 1024 * 8, 8.0f, 1.f);
	m_density->clear();
//	m_density.initialize(m_sceneDesc.aabb, m_outputBuffer.get_num_pixels() * m_params.maxPathLength * 2);
#endif
	// Automatically compute the splitting factor for photons.
	m_targetFlux = m_params.targetFlux * m_sceneDesc.lightTree.get_flux() / m_outputBuffer.get_num_pixels();

	logInfo("[NEB] View-Vertex map size: ", m_viewVertexMapManager.mem_size() / (1024*1024), " MB");
#ifdef NEB_KDTREE
	logInfo("[NEB] Density kd-tree size: ", m_density.mem_size() / (1024*1024), " MB");
#else
	logInfo("[NEB] Density octree size: ", m_density->mem_size() / (1024*1024), " MB");
#endif
	logInfo("[NEB] Self emission size: ", (m_selfEmissiveEndVertices.size() * sizeof(EmissionDesc)) / (1024*1024), " MB");
}

void CpuNextEventBacktracking::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer
