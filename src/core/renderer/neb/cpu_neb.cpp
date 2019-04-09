﻿#include "cpu_neb.hpp"
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
	//return 1.0f;
	return float(pdf) / (10.0f + float(pdf));
}
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
	i16 pathLen;
	float density { -1.0f };
	float prevRelativeProbabilitySum { 0.0f };
	float incidentDist;
	float neeConversion;	// Partial evaluation of the relPdf for the next event: (cosθ / d²) / nee.creationPdf

	void init(const PathVertex<NebVertexExt>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const AreaPdf incidentPdf, const float incidentCosineAbs,
			  const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput.weight;
		this->incidentDist = incidentDistance;
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
	VertexSample sample(const scene::materials::Medium* media,
						const math::RndSet2_1& rndSet,
						bool adjoint) const {
		VertexSample s = PathVertex<NebVertexExt>::sample(media, rndSet, adjoint);
		// Conditionally kill the photon path tracing if we are on the
		// NEE vertex. Goal: trace only photons which we cannot NEE.
		if(adjoint && ext().pathLen == 1) {
			float keepChance = get_photon_path_chance(s.pdf.forw);
			if(keepChance < ext().rnd) {
				s.type = math::PathEventType::INVALID;
			} else {
				s.pdf.forw *= keepChance;
				s.throughput /= keepChance;
			}
		}
		// HACK (energy clamping -- almost useless)
		//float energy = max(s.throughput) * 0.5f;
		//if(energy > 1.0f) s.throughput /= energy;
		return s;
	}
};

namespace {

int get_photon_split_count(const NebPathVertex& vertex, float maxFlux, const NebParameters& params) {
	float smoothness = ei::min(1e30f, vertex.get_pdf_max() * ei::PI);
	if(smoothness < 1e-3f) // Max-pdf cannot be that small (except the surface does not reflect at all)
		return 0;
	return ei::ceil(maxFlux / (smoothness * params.targetFlux));
}

struct { float toFlux; int photonCount; }
get_photon_conversion_factors(const NebPathVertex& vertex,
							  const NebVertexExt& ext, const NebParameters& params) {
	float toFlux = ei::abs(vertex.get_geometrical_factor(ext.neeDirection)) / ext.density;
	float smoothness = ei::min(1e30f, vertex.get_pdf_max() * ei::PI);
	float maxFlux = max(ext.neeIrradiance) * toFlux;
	int photonCount = get_photon_split_count(vertex, maxFlux, params);
	return { toFlux / ei::max(1,photonCount), photonCount };
}

float get_previous_merge_sum(const NebPathVertex& vertex, AngularPdf pdfBack) {
	if(!vertex.previous())
		return 0.0f;
	AreaPdf reversePdf = vertex.previous()->convert_pdf(Interaction::SURFACE, pdfBack,
		{ vertex.get_incident_direction(), ei::sq(vertex.ext().incidentDist) }).pdf;
	float relPdf = reversePdf / vertex.ext().incidentPdf;
	return relPdf + relPdf * vertex.previous()->ext().prevRelativeProbabilitySum;
}

float get_previous_merge_sum(const CpuNextEventBacktracking::PhotonDesc& photon, AngularPdf pdfBack) {
	AreaPdf reversePdf { photon.prevConversionFactor * float(pdfBack) };
	float relPdf = reversePdf / photon.incidentPdf;
	return relPdf + relPdf * photon.prevPrevRelativeProbabilitySum;
}
} // namespace ::

// Evaluate direct hit of area lights
CpuNextEventBacktracking::EmissionDesc
CpuNextEventBacktracking::evaluate_self_radiance(const NebPathVertex& vertex,
												 bool includeThroughput) {
	auto& guideFunction = m_params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													   : scene::lights::guide_flux;

	if(vertex.ext().pathLen >= m_params.minPathLength) {
		auto emission = vertex.get_emission();
		if(includeThroughput) emission.value *= vertex.ext().throughput;
		if(emission.value != 0.0f && vertex.ext().pathLen > 1) {
			AreaPdf startPdf = connect_pdf(m_sceneDesc.lightTree, vertex.get_primitive_id(),
			                               vertex.get_surface_params(),
			                               vertex.previous()->get_position(), guideFunction);
			// Get the NEE versus random hit chance.
			float relPdf = startPdf / vertex.ext().incidentPdf;
			float relSum = relPdf;
			scene::Direction incident = (vertex.get_position() - vertex.previous()->get_position()) / vertex.ext().incidentDist;
			float LtoE = ei::abs(vertex.get_geometrical_factor(incident) * vertex.previous()->get_geometrical_factor(incident))
				/ (float(emission.pdf) * ei::sq(vertex.ext().incidentDist));
			return { static_cast<const NebPathVertex*>(vertex.previous()), emission.value, relSum, LtoE };
		}
		return { nullptr, emission.value, 0.0f, 0.0f };
	}
	return { nullptr, Spectrum { 0.0f }, 0.0f, 0.0f };
}

// Evaluate a hit of the background
// TODO: unify by making the background behaving like an area light.
CpuNextEventBacktracking::EmissionDesc
CpuNextEventBacktracking::evaluate_background(const NebPathVertex& vertex, const VertexSample& sample, int pathLen) {
	if(pathLen >= m_params.minPathLength) {
		auto background = scene::lights::evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
		background.value *= vertex.ext().throughput;
		if(any(greater(background.value, 0.0f)) && pathLen > 1) {
			AreaPdf startPdf = background_pdf(m_sceneDesc.lightTree, background);
			// Get the NEE versus random hit chance.
			float relPdf = float(startPdf) / float(sample.pdf.forw);
			float relSum = relPdf;
			float LtoE = ei::abs(vertex.previous()->get_geometrical_factor(sample.excident))
				/ float(background.pdf.back);
			return { static_cast<const NebPathVertex*>(vertex.previous()), background.value, relSum, LtoE };
		}
		return { nullptr, background.value, 0.0f, 0.0f };
	}
	return { nullptr, Spectrum { 0.0f }, 0.0f, 0.0f };
}

void CpuNextEventBacktracking::sample_view_path(const Pixel coord, const int pixelIdx) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	NebPathVertex* previous = nullptr;
	NebPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	NebPathVertex::create_camera(&vertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[pixelIdx].next());

	auto& guideFunction = m_params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													   : scene::lights::guide_flux;

	int pathLen = 0;
	do {
		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { m_rngs[pixelIdx].next(), m_rngs[pixelIdx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[pixelIdx].next()));
		NebPathVertex& sourceVertex = previous ? *previous : vertex;	// Make sure the update function is called for the correct vertex.
		if(!walk(m_sceneDesc, sourceVertex, rnd, rndRoulette, false, throughput, vertex, sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto emission = evaluate_background(vertex, sample, pathLen+1);
				if(emission.radiance != 0.0f) {
					// In the case that there is no previous, there is also no MIS and we
					// contribute directly.
					if(!previous) {
						m_outputBuffer.contribute(coord, { Spectrum{1.0f}, 1.0f }, { Spectrum{1.0f}, 1.0f },
												  1.0f, emission.radiance);
					} else {
						u32 idx = m_selfEmissionCount.fetch_add(1);
						m_selfEmissiveEndVertices[idx] = emission;
					}
				}
			}
			break;
		}
		++pathLen;

		vertex.ext().pixelIndex = pixelIdx;
		vertex.ext().pathLen = pathLen;
		if(pathLen == m_params.maxPathLength) {
			// Using NEE on random hit vertices has low contribution in general.
			// Therefore, we store the preevaluated emission only.
			auto emission = evaluate_self_radiance(vertex, true);
			if(emission.radiance != 0.0f) {
				// In the case that there is no previous, there is also no MIS and we
				// contribute directly.
				if(!previous) {
					m_outputBuffer.contribute(coord, { Spectrum{1.0f}, 1.0f }, { Spectrum{1.0f}, 1.0f },
											  1.0f, emission.radiance);
				} else {
					u32 idx = m_selfEmissionCount.fetch_add(1);
					m_selfEmissiveEndVertices[idx] = emission;
				}
			}
		} else {
			// Simulate an NEE, but do not contribute. Instead store the resulting
			// vertex for later use.
			u64 neeSeed = m_rngs[pixelIdx].next();
			math::RndSet2 neeRnd = m_rngs[pixelIdx].next();
			auto nee = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
								vertex.get_position(), m_sceneDesc.aabb, neeRnd,
								guideFunction);
			if(nee.cosOut != 0) nee.diffIrradiance *= nee.cosOut;
			if(m_params.secondaryNEEs) {
				float tracingDist = (nee.dist >= scene::MAX_SCENE_SIZE) ? m_sceneDesc.diagSize : nee.dist;
				scene::Point neePos = vertex.get_position() + nee.direction * tracingDist;
				auto hit = scene::accel_struct::first_intersection(m_sceneDesc,
														{ neePos, -nee.direction },
														{}, tracingDist);
				if(hit.hitT < tracingDist * 0.999f) {
					// Hit a different surface than the current one.
					// Additionally storing this vertex further reduces variance for direct lighted
					// surfaces and allows the light-bulb scenario when using the backtracking.
					scene::Point hitPos = neePos - nee.direction * hit.hitT;
					const scene::TangentSpace tangentSpace = scene::accel_struct::tangent_space_geom_to_shader(m_sceneDesc, hit);
					NebPathVertex virtualLight;
					NebPathVertex::create_surface(&virtualLight, nullptr, hit,
						m_sceneDesc.get_material(hit.hitId), hitPos, tangentSpace, -nee.direction,
						hit.hitT, AngularPdf{1.0f}, math::Throughput{});
					// An additional NEE is necessary, such that this new vertex is an unbiased
					// estimate again.
					neeSeed = m_rngs[pixelIdx].next();
					neeRnd = m_rngs[pixelIdx].next();
					auto neeSec = connect(m_sceneDesc.lightTree, 0, 1, neeSeed,
										  hitPos, m_sceneDesc.aabb, neeRnd,
										  guideFunction);
					bool anyhit = scene::accel_struct::any_intersection(
										m_sceneDesc, { hitPos, neeSec.direction },
										hit.hitId, neeSec.dist);
					if(anyhit) neeSec.diffIrradiance = Spectrum{0.0f};
					if(nee.cosOut != 0) neeSec.diffIrradiance *= neeSec.cosOut;
					virtualLight.ext().pathLen = -32000;	// Mark this as non contributing (not connected to a pixel)
					virtualLight.ext().neeIrradiance = neeSec.diffIrradiance;
					virtualLight.ext().neeDirection = neeSec.direction;
					virtualLight.ext().neeConversion = neeSec.cosOut / (neeSec.distSq * float(neeSec.creationPdf));
					m_viewVertexMap.insert(hitPos, virtualLight);
					m_density.increment(hitPos);
					// Make sure the vertex for which we did the NEE knows it is shadowed.
					nee.diffIrradiance = Spectrum{0.0f};
				}
			} else {
				bool anyhit = scene::accel_struct::any_intersection(
									m_sceneDesc, { vertex.get_position(), nee.direction },
									vertex.get_primitive_id(), nee.dist);
				// Make sure the vertex for which we did the NEE knows it is shadowed.
				if(anyhit)
					nee.diffIrradiance = Spectrum{0.0f};
			}
			vertex.ext().neeIrradiance = nee.diffIrradiance;
			vertex.ext().neeDirection = nee.direction;
			vertex.ext().neeConversion = nee.cosOut / (nee.distSq * float(nee.creationPdf));
			previous = m_viewVertexMap.insert(vertex.get_position(), vertex);
			if(previous == nullptr) break;	// OVERFLOW
			m_density.increment(vertex.get_position());
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
	//vertex.ext().neeConversion /= count;
}

// Create the backtracking path
void CpuNextEventBacktracking::sample_photon_path(float neeMergeArea, float photonMergeArea, math::Rng& rng, const NebPathVertex& vertex) {
	if(vertex.ext().neeIrradiance == 0.0f) return;
	// Precomupte the irradiance -> flux factor and
	// pplit photons on rough surfaces if their density is much smaller than necessary
	auto pFactors = get_photon_conversion_factors(vertex, vertex.ext(), m_params);
	for(int i = 0; i < pFactors.photonCount; ++i) {
		// Prepare a start vertex to begin the sampling of the photon event.
		NebPathVertex virtualLight = vertex;
		virtualLight.set_incident_direction(-vertex.ext().neeDirection);

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
			virtualLight.ext().pathLen = lightPathLength;
			if(!walk(m_sceneDesc, virtualLight, rnd, rndRoulette, true, lightThroughput, virtualLight, sample))
				break;
			++lightPathLength;

			// Compute mis-weight partial sums.
			float prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / ei::sq(virtualLight.ext().incidentDist);
			float relPdfSum = 0.0f;
			if(previous) {
				float relPdf = previous->prevConversionFactor * float(sample.pdf.back) / float(previous->incidentPdf);
				relPdfSum = relPdf + relPdf * previous->prevPrevRelativeProbabilitySum;
			} else {
				// No previous photon means that the previous vertex was the NEE start vertex.
				// Compute the random hit event probability relative to this start vertex.
				float neeReuseCount = ei::max(1.0f, neeMergeArea * vertex.ext().density);
				relPdfSum = float(sample.pdf.back) * vertex.ext().neeConversion / neeReuseCount;
				// Additionally, the next vertex has to see the nee-merge with a respective factor.
				float photonReuseCount = photonMergeArea * vertex.ext().density * pFactors.photonCount;
				prevConversionFactor *= neeReuseCount / photonReuseCount;
			}

			// Store the new photon
			previous = m_photonMap.insert(virtualLight.get_position(),
				{ virtualLight.get_position(), virtualLight.ext().incidentPdf,
					sample.excident, lightPathLength,
					lightThroughput.weight * vertex.ext().neeIrradiance * pFactors.toFlux, relPdfSum,
					virtualLight.get_geometric_normal(), prevConversionFactor
				});
			if(previous == nullptr) break;	// OVERFLOW
			prevNormal = virtualLight.get_geometric_normal();
		}
	}
}

Spectrum CpuNextEventBacktracking::merge_photons(float mergeRadiusSq, const NebPathVertex& vertex) {
	scene::Point currentPos = vertex.get_position();
	Spectrum radiance { 0.0f };
	// Merge photons
	auto photonIt = m_photonMap.find_first(currentPos);
	while(photonIt) {
		auto& photon = *photonIt;
		int pathLen = photon.pathLen + vertex.ext().pathLen;
		if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
			&& lensq(photon.position - currentPos) < mergeRadiusSq) {
			Pixel tmpCoord;
			auto bsdf = vertex.evaluate(-photon.incident,
										m_sceneDesc.media, tmpCoord, false,
										&photon.geoNormal);
			// MIS compare against previous merges (view path) AND feature merges (light path)
			float relSum = get_previous_merge_sum(vertex, bsdf.pdf.back);
			relSum += get_previous_merge_sum(photon, bsdf.pdf.forw);
			float misWeight = 1.0f / (1.0f + relSum);
			radiance += bsdf.value * photon.irradiance * misWeight;
		}
		++photonIt;
	}
	return radiance / (ei::PI * mergeRadiusSq);
}

Spectrum CpuNextEventBacktracking::evaluate_nee(const NebPathVertex& vertex,
												const NebVertexExt& ext,
												float neeReuseCount, float photonReuseCount) {
	Pixel tmpCoord;
	auto bsdf = vertex.evaluate(ext.neeDirection,
								m_sceneDesc.media, tmpCoord, false,
								nullptr);
	// MIS compares against all previous merges (there are no feature ones)
	float relSum = get_previous_merge_sum(vertex, bsdf.pdf.back);
	relSum *= get_photon_path_chance(bsdf.pdf.back) * photonReuseCount / neeReuseCount;
	// And the random hit connection
	float relHitPdf = float(bsdf.pdf.forw) * ext.neeConversion / neeReuseCount;
	float misWeight = 1.0f / (1.0f + relSum + relHitPdf);
	return (misWeight * bsdf.cosOut) * bsdf.value * ext.neeIrradiance;
}

Spectrum CpuNextEventBacktracking::merge_nees(float mergeRadiusSq, float photonMergeArea, const NebPathVertex& vertex) {
	scene::Point currentPos = vertex.get_position();
	int neePathLen = vertex.ext().pathLen + 1;
	float photonReuseCount = photonMergeArea * vertex.ext().density;
	// If path length is already too large there will be no contribution from this vertex.
	// It only exists for the sake of random hit evaluation (and as additional sample).
	if(neePathLen >= m_params.minPathLength && neePathLen <= m_params.maxPathLength) {
		if(mergeRadiusSq == 0.0f) {
			// Merges are disabled -> use the current vertex only
			if(any(greater(vertex.ext().neeIrradiance, 0.0f))) {
				int photonCount = get_photon_conversion_factors(vertex, vertex.ext(), m_params).photonCount;
				return evaluate_nee(vertex, vertex.ext(), 1.0f, photonReuseCount * photonCount);
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
						radiance += evaluate_nee(vertex, otherExt, reuseCount, photonReuseCount * photonCount);
					}
				}
				++otherEndIt;
			}
			return radiance / count;
		}
	}
	return Spectrum { 0.0f };
}

Spectrum CpuNextEventBacktracking::finalize_emission(float neeMergeArea, float photonMergeArea, const EmissionDesc& emission) {
	float relSum = emission.relPdf;
	if(emission.previous) {
		// Compute the irradiance (max) at the previous vertex when illuminated by the current point (precomputed).
		// Then convert it further to flux by division with  the density
		float maxFlux = emission.radianceToIrradiance * max(emission.radiance) / emission.previous->ext().density;
		float neeReuseCount = ei::max(1.0f, neeMergeArea * emission.previous->ext().density);
		float photonReuseCount = photonMergeArea * emission.previous->ext().density
			* get_photon_split_count(*emission.previous, maxFlux, m_params);
		relSum = emission.relPdf * neeReuseCount
				+ emission.relPdf * emission.previous->ext().prevRelativeProbabilitySum
				* get_photon_path_chance(emission.previous->ext().pdfBack) * photonReuseCount;
	}
	float misWeight = 1.0f / (1.0f + relSum);
	return emission.radiance * misWeight;
}

CpuNextEventBacktracking::CpuNextEventBacktracking() {
}

void CpuNextEventBacktracking::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU NEB iteration", ProfileLevel::HIGH);

	float photonMergeRadiusSq = ei::sq(m_params.mergeRadius * m_sceneDesc.diagSize);
	float photonMergeArea = photonMergeRadiusSq * ei::PI;
	float neeMergeRadiusSq = ei::sq(m_params.neeMergeRadius * m_sceneDesc.diagSize);
	float neeMergeArea = neeMergeRadiusSq * ei::PI;
	m_viewVertexMap.clear(m_params.neeMergeRadius * m_sceneDesc.diagSize * 2.0001f);
	m_photonMap.clear(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);
	m_selfEmissionCount.store(0);
	m_density.set_iteration(m_currentIteration + 1);

	auto& guideFunction = m_params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													   : scene::lights::guide_flux;

	// First pass: distribute and store view path vertices.
	// For each vertex compute the next event estimate, but do not contribute yet.
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		Pixel coord { pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		sample_view_path(coord, pixel);
	}

	// Second pass: merge NEEs and backtrack. For each stored vertex find all other
	// vertices in the neighborhood and average the NEE result. Then trace a light path
	// beginning with a virtual source using the NEE direction as incident direction.
	// Store the new vertices in a standard photon map.
	i32 numViewVertices = m_viewVertexMap.size();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		//estimate_density(photonMergeRadiusSq, vertex);
		vertex.ext().density = m_density.get_density(vertex.get_position(), vertex.get_geometric_normal());
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
		if(vertex.ext().pathLen < 0) continue;
		scene::Point currentPos = vertex.get_position();
		Spectrum radiance { 0.0f };

		if(photonMergeArea > 0.0f)
			radiance += merge_photons(photonMergeRadiusSq, vertex);
		radiance += merge_nees(neeMergeRadiusSq, photonMergeArea, vertex);
		auto emission = evaluate_self_radiance(vertex, false);
		radiance += finalize_emission(neeMergeArea, photonMergeArea, emission);

		Pixel coord { vertex.ext().pixelIndex % m_outputBuffer.get_width(),
					  vertex.ext().pixelIndex / m_outputBuffer.get_width() };
		m_outputBuffer.contribute(coord, { vertex.ext().throughput, 1.0f }, { Spectrum{1.0f}, 1.0f },
								  1.0f, radiance);
	}

	// Finialize the evaluation of emissive end vertices.
	// It is necessary to do this after the density estimate for a correct mis.
	i32 selfEmissionCount = m_selfEmissionCount.load();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < selfEmissionCount; ++i) {
		Spectrum emission = finalize_emission(neeMergeArea, photonMergeArea, m_selfEmissiveEndVertices[i]);
		Pixel coord { m_selfEmissiveEndVertices[i].previous->ext().pixelIndex % m_outputBuffer.get_width(),
					  m_selfEmissiveEndVertices[i].previous->ext().pixelIndex / m_outputBuffer.get_width() };
		m_outputBuffer.contribute(coord, { Spectrum{1.0f}, 1.0f }, { Spectrum{1.0f}, 1.0f },
								  1.0f, emission);
	}//*/

	Profiler::instance().create_snapshot_all();

	logPedantic("[NEB] Memory occupation | View-Vertices: ", m_viewVertexMap.size() * 100.0f / float(m_viewVertexMap.capacity()),
				"% | Photons: ", m_photonMap.size() * 100.0f / float(m_photonMap.capacity()),
				"% | Octree: ", m_density.size() * 100.0f / float(m_density.capacity()), "%.");
}

void CpuNextEventBacktracking::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	//int countHeuristic = m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1) * 2; // Save count
	int countHeuristic = m_outputBuffer.get_num_pixels() * ei::ceil(logf(float(m_params.maxPathLength)) * 4.0f);
	m_viewVertexMapManager.resize(countHeuristic);
	m_viewVertexMap = m_viewVertexMapManager.acquire<Device::CPU>();
	m_photonMapManager.resize(countHeuristic / 2);
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
	// There is at most one emissive end vertex per path
	m_selfEmissiveEndVertices.resize(m_outputBuffer.get_num_pixels());
	m_density.initialize(m_sceneDesc.aabb, m_outputBuffer.get_num_pixels() * m_params.maxPathLength * 2);

	logInfo("[NEB] View-Vertex map size: ", m_viewVertexMapManager.mem_size() / (1024*1024), " MB");
	logInfo("[NEB] Photon map size: ", m_photonMapManager.mem_size() / (1024*1024), " MB");
	logInfo("[NEB] Density octree size: ", m_density.mem_size() / (1024*1024), " MB");
	logInfo("[NEB] Self emission size: ", (m_selfEmissiveEndVertices.size() * sizeof(EmissionDesc)) / (1024*1024), " MB");
	logInfo("[NEB] sizeof(PhotonDesc)=", sizeof(PhotonDesc), ", sizeof(NebPathVertex)=", sizeof(NebPathVertex));
}

void CpuNextEventBacktracking::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer