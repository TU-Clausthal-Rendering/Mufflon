#include "cpu_neb.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
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
float get_density(const scene::accel_struct::KdTree<char, 3>& tree, const ei::Vec3& currentPos) {
	int idx[4];
	float distSq[4];
	tree.query_euclidean(currentPos, 4, idx, distSq);
	float queryArea = (distSq[3]) * ei::PI / 3.0f;
	//float queryArea = (distSq[3] - distSq[2]) * ei::PI / 1.0f;
	return 1.0f / queryArea;
}
#endif

} // namespace ::


struct NebVertexExt {
	AngularPdf pdfBack;
	AreaPdf incidentPdf { 0.0f };
	Spectrum throughput;
	Spectrum neeIrradiance;
	union {
		int pixelIndex;
		float rnd;		// An additional random value (first vertex of light paths only).
	};
	scene::Direction neeDirection;
	float density { -1.0f };
	float prevRelativeProbabilitySum { 0.0f };
	float incidentDist;
	float neeConversion;	// Partial evaluation of the relPdf for the next event: (cosθ / d²)
	AreaPdf neeCreationPdf;
	AreaPdf neeBackPdf;		// neeSamplePdf * cosθs / d²

	CUDA_FUNCTION void init(const PathVertex<NebVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		this->throughput = Spectrum{1.0f};
	}

	CUDA_FUNCTION void update(const PathVertex<NebVertexExt>& prevVertex,
							  const PathVertex<NebVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
		this->throughput = throughput.weight;
		this->incidentDist = incident.distance;
	}

	void update(const PathVertex<NebVertexExt>& thisVertex,
				const scene::Direction& excident,
				const math::PdfPair& pdf) {
		pdfBack = pdf.back;
		const PathVertex<NebVertexExt>* prev = thisVertex.previous();
		if(prev) {
			AreaPdf reversePdf = prev->convert_pdf(Interaction::SURFACE, pdf.back,
				{ thisVertex.get_incident_direction(), ei::sq(thisVertex.ext().incidentDist) }).pdf;
			float relPdf = reversePdf / thisVertex.ext().incidentPdf;
			prevRelativeProbabilitySum = relPdf + relPdf * prev->ext().prevRelativeProbabilitySum;
		}
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

int get_photon_split_count(const NebPathVertex& vertex, float maxFlux, const NebParameters& params) {
	return 1; // Disabled (MIS not yet capable again)
	float smoothness = ei::min(1e30f, vertex.get_pdf_max() * ei::PI);
	if(smoothness < 1e-3f) // Max-pdf cannot be that small (except the surface does not reflect at all)
		return 0;
	return ei::ceil(maxFlux / (smoothness * params.targetFlux));
}

struct PhotonConversionFactors { float toFlux; int photonCount; };
PhotonConversionFactors
get_photon_conversion_factors(const NebPathVertex& vertex,
							  const NebVertexExt& ext, const NebParameters& params) {
	float toFlux = ei::abs(vertex.get_geometric_factor(ext.neeDirection)) / ext.density;
	float smoothness = ei::min(1e30f, vertex.get_pdf_max() * ei::PI);
	float maxFlux = max(ext.neeIrradiance) * toFlux;
	int photonCount = get_photon_split_count(vertex, maxFlux, params);
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
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
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
				m_outputBuffer.contribute(coord, { Spectrum{1.0f}, 1.0f }, { Spectrum{1.0f}, 1.0f },
											1.0f, emission.value);
			} else if(emission.value != 0.0f) {
				EmissionDesc emDesc;
				emDesc.previous = static_cast<const NebPathVertex*>(vertex.previous());
				emDesc.radiance = emission.value * throughput.weight;
				emDesc.incidentPdf = vertex.ext().incidentPdf;
				emDesc.startPdf = pathLen == 1 ? AreaPdf{0.0f} : emission.connectPdf;
				// Special case: when photons are traced they use the emit pdf and not the connect pdf.
				// Fortunatelly, the emDesc.samplePdf is only used for the tracing events, so we can
				// include the difference between the two events in this PDF.
				emDesc.samplePdf = pathLen == 1 ? AngularPdf{0.0f} : emission.pdf * (emission.emitPdf / emission.connectPdf);
				emDesc.incident = sample.excident;
				emDesc.incidentDistSq = ei::sq(vertex.ext().incidentDist);
				u32 idx = m_selfEmissionCount.fetch_add(1);
				m_selfEmissiveEndVertices[idx] = emDesc;
			}
		}
		if(vertex.is_end_point()) break;

		vertex.ext().pixelIndex = pixelIdx;
		// Using NEE on random hit vertices has low contribution in general.
		// Therefore, we store the preevaluated emission only.
		if(pathLen < m_params.maxPathLength) {
			// Simulate an NEE, but do not contribute. Instead store the resulting
			// vertex for later use.
			u64 neeSeed = m_rngs[pixelIdx].next();
			math::RndSet2 neeRnd = m_rngs[pixelIdx].next();
			auto nee = scene::lights::connect(m_sceneDesc, 0, 1, neeSeed, vertex.get_position(), neeRnd);
			if(nee.cosOut != 0) nee.diffIrradiance *= nee.cosOut;
			bool anyhit = scene::accel_struct::any_intersection(
								m_sceneDesc, vertex.get_position(), nee.position,
								vertex.get_geometric_normal(), nee.geoNormal, nee.dir.direction);
			// Make sure the vertex for which we did the NEE knows it is shadowed.
			if(anyhit)
				nee.diffIrradiance = Spectrum{0.0f};
			vertex.ext().neeIrradiance = nee.diffIrradiance;
			vertex.ext().neeDirection = nee.dir.direction;
			vertex.ext().neeConversion = nee.cosOut / nee.distSq;
			vertex.ext().neeCreationPdf = nee.creationPdf;
			vertex.ext().neeBackPdf = nee.dir.pdf.to_area_pdf(ei::abs(vertex.get_geometric_factor(nee.dir.direction)), nee.distSq);
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

// Estimate the density around a vertex and apply the density to the NEE value
void CpuNextEventBacktracking::estimate_density(float densityEstimateRadiusSq, NebPathVertex& vertex) {
	scene::Point currentPos = vertex.get_position();
	int count = 0;
	// If path length is already too large there will be no contribution from this vertex.
	// It only exists for the sake of random hit evaluation (and as additional sample).
	auto otherEndIt = m_viewVertexMap.find_first(currentPos);
	while(otherEndIt) {
		if(lensq(otherEndIt->get_position() - currentPos) < densityEstimateRadiusSq) 
			++count;
		++otherEndIt;
	}
	mAssert(count >= 1);
	vertex.ext().density = count / (ei::PI * densityEstimateRadiusSq);
}

// Create the backtracking path
void CpuNextEventBacktracking::sample_photon_path(float neeMergeArea, float photonMergeArea, math::Rng& rng, const NebPathVertex& vertex) {
	if(vertex.ext().neeIrradiance == 0.0f) return;
	// Precomupte the irradiance -> flux factor and
	// pplit photons on rough surfaces if their density is much smaller than necessary
	auto pFactors = get_photon_conversion_factors(vertex, vertex.ext(), m_params);
	Spectrum flux = vertex.ext().neeIrradiance * pFactors.toFlux;
	// Clamping for bad photons (due to errors in the density estimation)
	//float expectedFluxMax = max(flux);
	//flux *= ei::min(expectedFluxMax, m_params.targetFlux) / expectedFluxMax;
	for(int i = 0; i < pFactors.photonCount; ++i) {
		// Prepare a start vertex to begin the sampling of the photon event.
		NebPathVertex virtualLight = vertex;
		virtualLight.set_incident_direction(-vertex.ext().neeDirection, nullptr);

		// Trace a path
		virtualLight.ext().rnd = math::sample_uniform(u32(rng.next()));
		int lightPathLength = 1;
		math::Throughput lightThroughput { Spectrum{1.0f}, 0.0f };
		scene::Direction prevNormal = virtualLight.get_geometric_normal();
		CpuNextEventBacktracking::PhotonDesc* previous = nullptr;
		while(lightPathLength < m_params.maxPathLength-1) {
			math::RndSet2_1 rnd { rng.next(), rng.next() };
			float rndRoulette = math::sample_uniform(u32(rng.next()));
			VertexSample sample;
			if(walk(m_sceneDesc, virtualLight, rnd, rndRoulette, true, lightThroughput, virtualLight, sample) != WalkResult::HIT)
				break;
			++lightPathLength;

			PhotonDesc photon;
			if(previous) {
				photon.previous = previous;
				previous->pdfBack = sample.pdf.back;	// Set the previously unknown backward pdf
			} else {
				photon.prev.creationPdf = vertex.ext().neeCreationPdf;
				photon.prev.incidentPdf = vertex.ext().neeBackPdf;
				photon.prev.hitPdf = AreaPdf{float(sample.pdf.back) * vertex.ext().neeConversion};
			}
			photon.position = virtualLight.get_position();
			photon.incidentPdf = virtualLight.ext().incidentPdf;
			photon.incident = sample.excident;
			photon.pathLen = lightPathLength;
			photon.flux = lightThroughput.weight * flux;
			photon.geoNormal = virtualLight.get_geometric_normal();
			photon.prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / ei::sq(virtualLight.ext().incidentDist);
			photon.pdfBack = AngularPdf{0.0f}; // Unknown (is set in the next iteration)
			photon.sourceDensity = vertex.ext().density;

			// Store the new photon
			previous = m_photonMap.insert(virtualLight.get_position(), photon);
			if(previous == nullptr) { mAssert(false); break; }	// OVERFLOW
			prevNormal = virtualLight.get_geometric_normal();
		}
	}
}

void CpuNextEventBacktracking::sample_std_photon(int idx, int numPhotons, u64 seed, float photonMergeArea) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	//u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	NebPathVertex vertex[2];
	NebPathVertex::create_light(&vertex[0], nullptr, p);
	math::Throughput throughput;
	scene::Direction prevNormal = vertex[0].get_geometric_normal();

	float prevPrevConversionFactor = 0.0f;
	float relPdfSum = 0.0f;
	float mergeConversionFactor = 0.0f;
	int lightPathLength = 0;
	int currentV = 0;
	CpuNextEventBacktracking::PhotonDesc* previous = nullptr;
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

		PhotonDesc photon;
		if(previous) {
			photon.previous = previous;
			previous->pdfBack = sample.pdf.back;	// Set the previously unknown backward pdf
			photon.sourceDensity = previous->sourceDensity;
		} else {
			photon.prev.creationPdf = vertex[0].ext().incidentPdf;
#ifdef NEB_KDTREE
			photon.sourceDensity = get_density(m_density, vertex[currentV].get_position());
#else
			photon.sourceDensity = m_density->get_density_interpolated(vertex[currentV].get_position(), vertex[currentV].get_geometric_normal());
#endif
		}
		photon.position = vertex[currentV].get_position();
		photon.incidentPdf = vertex[currentV].ext().incidentPdf;
		photon.incident = sample.excident;
		photon.pathLen = -lightPathLength;
		photon.flux = throughput.weight / numPhotons;
		photon.geoNormal = vertex[currentV].get_geometric_normal();
		photon.prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / ei::sq(vertex[currentV].ext().incidentDist);
		photon.pdfBack = AngularPdf{0.0f}; // Unknown (is set in the next iteration)

		// Store the new photon
		previous = m_photonMap.insert(vertex[currentV].get_position(), photon);
		if(previous == nullptr) break;	// OVERFLOW
		prevNormal = vertex[currentV].get_geometric_normal();
	}
}

Spectrum CpuNextEventBacktracking::merge_photons(float mergeRadiusSq, const NebPathVertex& vertex,
	AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons) {
	scene::Point currentPos = vertex.get_position();
	Spectrum radiance { 0.0f };
	// Merge photons
	auto photonIt = m_photonMap.find_first(currentPos);
	while(photonIt) {
		auto& photon = *photonIt;
		int pathLen = ei::abs(photon.pathLen) + vertex.get_path_len();
		if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
			&& lensq(photon.position - currentPos) < mergeRadiusSq) {
			Pixel tmpCoord;
			auto bsdf = vertex.evaluate(-photon.incident,
										m_sceneDesc.media, tmpCoord, false,
										&photon.geoNormal);
			if(any(greater(bsdf.value, 0.0f))) {
				const PhotonDesc* nextP = &photon;
				AngularPdf pdfForw = bsdf.pdf.forw;
				for(int i = vertex.get_path_len(); i < pathLen; ++i) {
					incidentF[i+1] = AreaPdf{float(pdfForw) * nextP->prevConversionFactor};
					incidentB[i] = nextP->incidentPdf;
					if(nextP->pathLen == 2) {
						incidentB[i+1] = nextP->prev.incidentPdf;
						incidentF[i+2] = nextP->prev.hitPdf;
						incidentB[i+2] = nextP->prev.creationPdf;
						break;
					}
					if(nextP->pathLen == -1) {
						incidentB[i+1] = nextP->prev.creationPdf;
						break;
					}
					nextP = nextP->previous;
					pdfForw = nextP->pdfBack;
				}
				incidentF[vertex.get_path_len()] = vertex.ext().incidentPdf;
				AngularPdf pdfBack = bsdf.pdf.back;
				ConnectionDir connection { vertex.get_incident_direction(), ei::sq(vertex.ext().incidentDist) };
				const auto* vert = vertex.previous();
				for(int i = vertex.get_path_len() - 1; i > 0; --i) {
					incidentF[i] = vert->ext().incidentPdf;
					incidentB[i] = vert->convert_pdf(Interaction::SURFACE, pdfBack, connection).pdf;
					pdfBack = vert->ext().pdfBack;
					connection = ConnectionDir{ vert->get_incident_direction(), ei::sq(vert->ext().incidentDist) };
					vert = vert->previous();
				}
				incidentF[0] = AreaPdf{1.0f};
				incidentB[0] = AreaPdf{0.0f};
				float misWeight;
				if(photon.pathLen < 0)
					misWeight = mis_stdphoton(incidentF, incidentB, pathLen, vertex.get_path_len(),
						photon.sourceDensity, ei::PI * mergeRadiusSq, numPhotons);
				else
					misWeight = mis_nebphoton(incidentF, incidentB, pathLen, vertex.get_path_len(),
						photon.sourceDensity, ei::PI * mergeRadiusSq, numPhotons, m_params.stdPhotons);
				radiance += bsdf.value * photon.flux * misWeight;
				mAssert(!isnan(radiance.r));
			} // BSDF > 0
		} // Pathlen and photon distance
		++photonIt;
	}
	return radiance / (ei::PI * mergeRadiusSq);
}

Spectrum CpuNextEventBacktracking::evaluate_nee(const NebPathVertex& vertex,
												const NebVertexExt& ext,
												float neeReuseCount,
												AreaPdf* incidentF, AreaPdf* incidentB,
												int numPhotons, float photonMergeArea) {
	Pixel tmpCoord;
	auto bsdf = vertex.evaluate(ext.neeDirection,
								m_sceneDesc.media, tmpCoord, false,
								nullptr); // TODO: use light normal for merges
	// MIS compares against all previous merges (there are no feature ones)
	int n = vertex.get_path_len() + 1;
	incidentF[n] = AreaPdf{ float(bsdf.pdf.forw) * ext.neeConversion };
	incidentB[n] = ext.neeCreationPdf;
	incidentF[n-1] = vertex.ext().incidentPdf;
	incidentB[n-1] = ext.neeBackPdf;
	const auto* vert = vertex.previous();
	AngularPdf pdfBack = bsdf.pdf.back;
	ConnectionDir connection { vertex.get_incident_direction(), ei::sq(vertex.ext().incidentDist) };
	for(int i = n-2; i > 0; --i) {
		incidentF[i] = vert->ext().incidentPdf;
		incidentB[i] = vert->convert_pdf(Interaction::SURFACE, pdfBack, connection).pdf;
		pdfBack = vert->ext().pdfBack;
		connection = ConnectionDir{ vert->get_incident_direction(), ei::sq(vert->ext().incidentDist) };
		vert = vert->previous();
	}
	incidentF[0] = AreaPdf{1.0f};
	incidentB[0] = AreaPdf{0.0f};
	float misWeight = mis_nee(incidentF, incidentB, n, vertex.ext().density, photonMergeArea, numPhotons, m_params.stdPhotons);
	return (misWeight * bsdf.cosOut) * bsdf.value * ext.neeIrradiance;
}

Spectrum CpuNextEventBacktracking::merge_nees(float mergeRadiusSq, float photonMergeArea, const NebPathVertex& vertex,
	AreaPdf* incidentF, AreaPdf* incidentB, int numPhotons) {
	scene::Point currentPos = vertex.get_position();
	int neePathLen = vertex.get_path_len() + 1;
	float photonReuseCount = photonMergeArea * vertex.ext().density;
	// If path length is already too large there will be no contribution from this vertex.
	// It only exists for the sake of random hit evaluation (and as additional sample).
	if(neePathLen >= m_params.minPathLength && neePathLen <= m_params.maxPathLength) {
		if(mergeRadiusSq == 0.0f) {
			// Merges are disabled -> use the current vertex only
			if(any(greater(vertex.ext().neeIrradiance, 0.0f))) {
				int photonCount = get_photon_conversion_factors(vertex, vertex.ext(), m_params).photonCount;
				return evaluate_nee(vertex, vertex.ext(), 1.0f, incidentF, incidentB, numPhotons, photonMergeArea);
			}
		} else {
			Spectrum radiance { 0.0f };
			int count = 0;	// Number of merged NEE events
			float reuseCount = ei::max(1.0f, mergeRadiusSq * ei::PI * vertex.ext().density);
			auto otherEndIt = m_viewVertexMap.find_first(currentPos);
			while(otherEndIt) {
				auto& otherExt = otherEndIt->ext();
				if(lensq(otherEndIt->get_position() - currentPos) < mergeRadiusSq) {
					++count;
					if(any(greater(otherExt.neeIrradiance, 0.0f))) {
						int photonCount = get_photon_conversion_factors(vertex, otherEndIt->ext(), m_params).photonCount;
						radiance += evaluate_nee(vertex, otherExt, reuseCount, incidentF, incidentB, numPhotons, photonMergeArea);
					}
				}
				++otherEndIt;
			}
			return radiance / count;
		}
	}
	return Spectrum { 0.0f };
}

Spectrum CpuNextEventBacktracking::finalize_emission(float neeMergeArea, float photonMergeArea, const EmissionDesc& emission,
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
			connection = ConnectionDir{ vert->get_incident_direction(), ei::sq(vert->ext().incidentDist) };
			vert = vert->previous();
		}
		incidentF[0] = AreaPdf{1.0f};
		incidentB[0] = AreaPdf{0.0f};
		float misWeight = mis_rhit(incidentF, incidentB, n,
			emission.previous->ext().density, photonMergeArea, numPhotons, m_params.stdPhotons);
		return emission.radiance * misWeight;
	} else
		return emission.radiance;
}

CpuNextEventBacktracking::CpuNextEventBacktracking() {
}

void CpuNextEventBacktracking::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU NEB iteration", ProfileLevel::HIGH);

	m_sceneDesc.lightTree.posGuide = m_params.neeUsePositionGuide;

	float photonMergeRadiusSq = ei::sq(m_params.mergeRadius * m_sceneDesc.diagSize);
	float photonMergeArea = photonMergeRadiusSq * ei::PI;
	float neeMergeRadiusSq = ei::sq(m_params.neeMergeRadius * m_sceneDesc.diagSize);
	float neeMergeArea = neeMergeRadiusSq * ei::PI;
	m_viewVertexMap.clear(m_params.neeMergeRadius * m_sceneDesc.diagSize * 2.0001f);
	m_photonMap.clear(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
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

	// Additional standard photons
	int numStdPhotonPaths = m_outputBuffer.get_num_pixels();
	if(m_params.stdPhotons) {
#pragma PARALLEL_FOR
		for(int pixel = 0; pixel < numStdPhotonPaths; ++pixel) {
			sample_std_photon(pixel, numStdPhotonPaths, photonSeed, photonMergeArea);
		}
	}

	// Second pass: merge NEEs and backtrack. For each stored vertex find all other
	// vertices in the neighborhood and average the NEE result. Then trace a light path
	// beginning with a virtual source using the NEE direction as incident direction.
	// Store the new vertices in a standard photon map.
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
#ifdef NEB_KDTREE
		vertex.ext().density = get_density(m_density, vertex.get_position());
#else
		//vertex.ext().density = m_density.get_density(vertex.get_position(), vertex.get_geometric_normal());
		vertex.ext().density = m_density->get_density_interpolated(vertex.get_position(), vertex.get_geometric_normal());
#endif
		mAssert(vertex.ext().density < 1e38f);

		int rngIndex = i % m_outputBuffer.get_num_pixels();
		sample_photon_path(neeMergeArea, photonMergeArea, m_rngs[rngIndex], vertex);
	}//*/

	// Third pass: merge backtracked photons, average NEE events and compute random hit
	//		contributions.
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		// Secondary source vertices to not contribute (not connected to a pixel)
		if(vertex.ext().pixelIndex < 0) continue;
		scene::Point currentPos = vertex.get_position();
		Spectrum radiance { 0.0f };
		AreaPdf* incidentPdfsF = tmpPdfMem.data() + (m_params.maxPathLength+1)*2*get_current_thread_idx();
		AreaPdf* incidentPdfsB = incidentPdfsF + m_params.maxPathLength+1;

		if(photonMergeArea > 0.0f)
			radiance += merge_photons(photonMergeRadiusSq, vertex, incidentPdfsF, incidentPdfsB, numStdPhotonPaths);
		radiance += merge_nees(neeMergeRadiusSq, photonMergeArea, vertex, incidentPdfsF, incidentPdfsB, numStdPhotonPaths);//*/
		//scene::Point lastPos = vertex.previous() ? vertex.previous()->get_position() : {0.0f};
		//auto emission = vertex.get_emission(m_sceneDesc, lastPos);
		//radiance += finalize_emission(neeMergeArea, photonMergeArea, emission, incidentPdfsF, incidentPdfsB, numStdPhotonPaths);//*/

		Pixel coord { vertex.ext().pixelIndex % m_outputBuffer.get_width(),
					  vertex.ext().pixelIndex / m_outputBuffer.get_width() };
		m_outputBuffer.contribute(coord, { vertex.ext().throughput, 1.0f }, { Spectrum{1.0f}, 1.0f },
								  1.0f, radiance);
		/*if(vertex.get_path_len() == 1)
			m_outputBuffer.set(coord, 0, ei::Vec3{vertex.ext().density * (m_currentIteration+1)});//*/
	}

	// Finialize the evaluation of emissive end vertices.
	// It is necessary to do this after the density estimate for a correct mis.
	i32 selfEmissionCount = m_selfEmissionCount.load();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < selfEmissionCount; ++i) {
		AreaPdf* incidentPdfsF = tmpPdfMem.data() + (m_params.maxPathLength+1)*2*get_current_thread_idx();
		AreaPdf* incidentPdfsB = incidentPdfsF + m_params.maxPathLength+1;
		Spectrum emission = finalize_emission(neeMergeArea, photonMergeArea, m_selfEmissiveEndVertices[i], incidentPdfsF, incidentPdfsB, numStdPhotonPaths);
		Pixel coord { m_selfEmissiveEndVertices[i].previous->ext().pixelIndex % m_outputBuffer.get_width(),
					  m_selfEmissiveEndVertices[i].previous->ext().pixelIndex / m_outputBuffer.get_width() };
		m_outputBuffer.contribute(coord, { Spectrum{1.0f}, 1.0f }, { Spectrum{1.0f}, 1.0f },
								  1.0f, emission);
	}//*/

	logPedantic("[NEB] Memory occupation    View-Vertices: ", m_viewVertexMap.size() * 100.0f / float(m_viewVertexMap.capacity()),
				"% | Photons: ", m_photonMap.size() * 100.0f / float(m_photonMap.capacity()),
				"% | Octree: ", m_density->size() * 100.0f / float(m_density->capacity()), "%.");
#ifdef NEB_KDTREE
	logPedantic("[NEB] KD-Tree depth: ", m_density.compute_depth(), " optimal: ", ei::ilog2(m_density.size())+1);
#endif
}

void CpuNextEventBacktracking::post_reset() {
	ResetEvent resetFlags { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
							ResetEvent::ALL : get_reset_event() };
	init_rngs(m_outputBuffer.get_num_pixels());
	//int countHeuristic = m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1) * 2; // Save count
	int countHeuristic = m_outputBuffer.get_num_pixels() * ei::ceil(logf(float(m_params.maxPathLength)) * 4.0f);
	m_viewVertexMapManager.resize(countHeuristic);
	m_viewVertexMap = m_viewVertexMapManager.acquire<Device::CPU>();
	int photonHeuristic = countHeuristic / 2;
	if(m_params.stdPhotons) photonHeuristic *= 3;
	m_photonMapManager.resize(photonHeuristic);
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
	// There is at most one emissive end vertex per path
	m_selfEmissiveEndVertices.resize(m_outputBuffer.get_num_pixels());
#ifdef NEB_KDTREE
	m_density.reserve(countHeuristic);
#else
	if(resetFlags.geometry_changed())
		m_density = std::make_unique<data_structs::DmOctree>(m_sceneDesc.aabb,
			1024 * 1024 * 8, 8.0f, true);
	m_density->clear();
//	m_density.initialize(m_sceneDesc.aabb, m_outputBuffer.get_num_pixels() * m_params.maxPathLength * 2);
#endif

	logInfo("[NEB] View-Vertex map size: ", m_viewVertexMapManager.mem_size() / (1024*1024), " MB");
	logInfo("[NEB] Photon map size: ", m_photonMapManager.mem_size() / (1024*1024), " MB");
#ifdef NEB_KDTREE
	logInfo("[NEB] Density kd-tree size: ", m_density.mem_size() / (1024*1024), " MB");
#else
	logInfo("[NEB] Density octree size: ", m_density->mem_size() / (1024*1024), " MB");
#endif
	logInfo("[NEB] Self emission size: ", (m_selfEmissiveEndVertices.size() * sizeof(EmissionDesc)) / (1024*1024), " MB");
	logInfo("[NEB] sizeof(PhotonDesc)=", sizeof(PhotonDesc), ", sizeof(NebPathVertex)=", sizeof(NebPathVertex));
}

void CpuNextEventBacktracking::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer