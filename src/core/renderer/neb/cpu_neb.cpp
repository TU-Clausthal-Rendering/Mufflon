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

struct NebVertexExt {
	NebPathVertex* previous;
	//scene::Direction excident;
	//AngularPdf pdf;
	AreaPdf incidentPdf { 0.0f };
	Spectrum throughput;
	Spectrum neeIrradiance;
	Pixel coord;
	scene::Direction neeDirection;
	i16 pathLen;
	i16 count { -1 };
	float prevRelativeProbabilitySum { 0.0f };
	float incidentDist;
	float neeConversion;	// Partial evaluation of the relPdf for the next event: (cosθ / d²) / nee.creationPdf

	CUDA_FUNCTION void init(const NebPathVertex& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const AreaPdf incidentPdf, const float incidentCosineAbs,
			  const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput.weight;
		this->incidentDist = incidentDistance;
	}

	CUDA_FUNCTION void update(const NebPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		//excident = sample.excident;
		//pdf = sample.pdfF;
		const NebPathVertex* prev = thisVertex.ext().previous;
		if(prev) {
			AreaPdf reversePdf = prev->convert_pdf(Interaction::SURFACE, pdf.back,
				{ thisVertex.get_incident_direction(), ei::sq(thisVertex.ext().incidentDist) }).pdf;
			float relPdf = reversePdf / thisVertex.ext().incidentPdf;
			prevRelativeProbabilitySum = relPdf + relPdf * prev->ext().prevRelativeProbabilitySum;
		}
	}
};


// Create the backtracking path
void sample_photons(const scene::SceneDescriptor<CURRENT_DEV>& scene,
					const NebParameters& params, math::Rng& rng, NebPathVertex& virtualLight,
					HashGrid<Device::CPU, CpuNextEventBacktracking::PhotonDesc>& photonMap) {
	int lightPathLength = 1;
	math::Throughput lightThroughput { virtualLight.ext().neeIrradiance, 0.0f };
	math::RndSet2_1 rnd { rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	VertexSample sample;
	scene::Direction prevNormal = virtualLight.get_geometric_normal();
	CpuNextEventBacktracking::PhotonDesc* previous = nullptr;
	float neeConversion = virtualLight.ext().neeConversion / virtualLight.ext().count;
	while(lightPathLength < params.maxPathLength-1
		&& walk(scene, virtualLight, rnd, rndRoulette, true, lightThroughput, virtualLight, sample)) {
		++lightPathLength;
		float prevConversionFactor = ei::abs(dot(prevNormal, sample.excident)) / ei::sq(virtualLight.ext().incidentDist);
		float relPdfSum = 0.0f;
		if(previous) {
			float relPdf = previous->prevConversionFactor * float(sample.pdf.back) / float(previous->incidentPdf);
			relPdfSum = relPdf + relPdf * previous->prevRelativeProbabilitySum;
		} else {
			// No previous photon means that the previous vertex was the NEE start vertex.
			// Compute the random hit event probability relative to this start vertex.
			relPdfSum = float(sample.pdf.back) * neeConversion;
		}
		previous = photonMap.insert(virtualLight.get_position(),
			{ virtualLight.get_position(), virtualLight.ext().incidentPdf,
				sample.excident, lightPathLength,
				lightThroughput.weight, relPdfSum,
				virtualLight.get_geometric_normal(), prevConversionFactor
			});
		prevNormal = virtualLight.get_geometric_normal();
	}
}

void sample_view_path(const scene::SceneDescriptor<CURRENT_DEV>& scene,
					  const NebParameters& params,
					  const Pixel coord,
					  math::Rng& rng,
					  HashGrid<Device::CPU, NebPathVertex>& viewVertexMap,
					  HashGrid<Device::CPU, CpuNextEventBacktracking::PhotonDesc>& photonMap) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	NebPathVertex* previous = nullptr;
	NebPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	NebPathVertex::create_camera(&vertex, &vertex, scene.camera.get(), coord, rng.next());

	auto& guideFunction = params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													 : scene::lights::guide_flux;

	int pathLen = 0;
	do {
		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		NebPathVertex& sourceVertex = previous ? *previous : vertex;	// Make sure the update function is called for the correct vertex.
		if(!walk(scene, sourceVertex, rnd, rndRoulette, false, throughput, vertex, sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// TODO: store void photon?
				// Missed scene - sample background
				/*auto background = evaluate_background(scene.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					AreaPdf startPdf = background_pdf(scene.lightTree, background);
					float mis = 1.0f / (1.0f + params.neeCount * float(startPdf) / float(sample.pdf.forw));
					background.value *= mis;
					outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}*/
			}
			break;
		}
		++pathLen;

		// Simulate an NEE, but do not contribute. Instead store the resulting
		// vertex for later use.
		u64 neeSeed = rng.next();
		math::RndSet2 neeRnd = rng.next();
		auto nee = connect(scene.lightTree, 0, 1, neeSeed,
							vertex.get_position(), scene.aabb, neeRnd,
							guideFunction);
		if(nee.cosOut != 0) nee.diffIrradiance *= nee.cosOut;
		//if(any(greater(nee.diffIrradiance, 0.0f))) {
		scene::Point neePos = vertex.get_position() + nee.direction * nee.dist;
		auto hit = scene::accel_struct::first_intersection(scene,
												{ neePos, -nee.direction },
												{}, nee.dist);
		if(hit.hitT < nee.dist * 0.999f) {
			// Hit a different surface than the current one.
			// Additionally storing this vertex further reduces variance for direct lighted
			// surfaces and allows the light-bulb scenario when using the backtracking.
			/*scene::Point hitPos = neePos - nee.direction * hit.hitT;
			const scene::TangentSpace tangentSpace = scene::accel_struct::tangent_space_geom_to_shader(scene, hit);
			NebPathVertex virtualLight;
			NebPathVertex::create_surface(&virtualLight, &virtualLight, hit,
				scene.get_material(hit.hitId), hitPos, tangentSpace, -nee.direction,
				nee.dist, AngularPdf{1.0f}, math::Throughput{});
			// Compensate the changed distance in diffIrradiance.
			virtualLight.ext().neeIrradiance = nee.diffIrradiance * nee.distSq / (hit.hitT * hit.hitT);
			virtualLight.ext().neeDirection = nee.direction;
			virtualLight.ext().coord = coord;
			virtualLight.ext().pathLen = -100000;	// Mark this as non contributing (not connected to a pixel)
			viewVertexMap.insert(virtualLight.get_position(), virtualLight);*/
			// Make sure the vertex for which we did the NEE knows it is shadowed.
			nee.diffIrradiance *= 0.0f;
		}
		//}
		vertex.ext().neeIrradiance = nee.diffIrradiance;
		vertex.ext().neeDirection = nee.direction;
		vertex.ext().neeConversion = nee.cosOut / (nee.distSq * float(nee.creationPdf));
		vertex.ext().coord = coord;
		vertex.ext().pathLen = pathLen;
		vertex.ext().previous = previous;
		previous = viewVertexMap.insert(vertex.get_position(), vertex);
	} while(pathLen < params.maxPathLength);
}

float get_previous_merge_sum(const NebPathVertex& vertex, AngularPdf pdfBack) {
	if(!vertex.ext().previous)
		return 0.0f;
	AreaPdf reversePdf = vertex.ext().previous->convert_pdf(Interaction::SURFACE, pdfBack,
		{ vertex.get_incident_direction(), ei::sq(vertex.ext().incidentDist) }).pdf;
	float relPdf = reversePdf / vertex.ext().incidentPdf;
	return relPdf + relPdf * vertex.ext().previous->ext().prevRelativeProbabilitySum;
}

float get_previous_merge_sum(const CpuNextEventBacktracking::PhotonDesc& photon, AngularPdf pdfBack) {
	AreaPdf reversePdf { photon.prevConversionFactor * float(pdfBack) };
	float relPdf = reversePdf / photon.incidentPdf;
	return relPdf + relPdf * photon.prevRelativeProbabilitySum;
}

} // namespace ::

CpuNextEventBacktracking::CpuNextEventBacktracking() {
}

void CpuNextEventBacktracking::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU NEB iteration", ProfileLevel::HIGH);

	float sceneSize = len(m_sceneDesc.aabb.max - m_sceneDesc.aabb.min);
	float currentMergeRadius = m_params.mergeRadius * sceneSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeArea = mergeRadiusSq * ei::PI;
	m_viewVertexMap.clear(currentMergeRadius * 2.0001f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);

	auto& guideFunction = m_params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													   : scene::lights::guide_flux;

	// First pass: distribute and store view path vertices.
	// For each vertex compute the next event estimate, but do not contribute yet.
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		Pixel coord { pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		sample_view_path(m_sceneDesc, m_params, coord, m_rngs[pixel], m_viewVertexMap, m_photonMap);
	}

	// Second pass: merge NEEs and backtrack. For each stored vertex find all other
	// vertices in the neighborhood and average the NEE result. Then trace a light path
	// beginning with a virtual source using the NEE direction as incident direction.
	// Store the new vertices in a standard photon map.
	i32 numViewVertices = m_viewVertexMap.size();
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		scene::Point currentPos = vertex.get_position();
		int count = 0;
		// If path length is already too large there will be no contribution from this vertex.
		// It only exists for the sake of random hit evaluation (and as additional sample).
		auto otherEndIt = m_viewVertexMap.find_first(currentPos);
		while(otherEndIt) {
			if(lensq(otherEndIt->get_position() - currentPos) < mergeRadiusSq) 
				++count;
			++otherEndIt;
		}
		mAssert(count >= 1);
		vertex.ext().count = count;
		vertex.ext().neeIrradiance /= count;

		NebPathVertex virtualLight = vertex;
		virtualLight.set_incident_direction(-vertex.ext().neeDirection);
		float cosLight = ei::abs(vertex.get_geometrical_factor(vertex.ext().neeDirection));
		virtualLight.ext().neeIrradiance *= cosLight;
		if(virtualLight.ext().neeIrradiance != 0.0f) {
			int rngIndex = i % m_outputBuffer.get_num_pixels();
			sample_photons(m_sceneDesc, m_params, m_rngs[rngIndex], virtualLight, m_photonMap);
		}
	}

	// Third pass: merge backtracked photons, average NEE events and compute random hit
	//		contributions.
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
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

		// Merge other NEE events
		int neePathLen = vertex.ext().pathLen + 1;
		// If path length is already too large there will be no contribution from this vertex.
		// It only exists for the sake of random hit evaluation (and as additional sample).
		if(neePathLen >= m_params.minPathLength && neePathLen <= m_params.maxPathLength) {
			auto otherEndIt = m_viewVertexMap.find_first(currentPos);
			while(otherEndIt) {
				auto& otherExt = otherEndIt->ext();
				if((otherExt.neeIrradiance != 0.0f)
				&& (lensq(otherEndIt->get_position() - currentPos) < mergeRadiusSq) ) {
					Pixel tmpCoord;
					auto bsdf = vertex.evaluate(otherExt.neeDirection,
												m_sceneDesc.media, tmpCoord, false,
												nullptr);
					// MIS compares against all previous merges (there are no feature ones)
					float relSum = get_previous_merge_sum(vertex, bsdf.pdf.back);
					// And the random hit connection
					float relHitPdf = float(bsdf.pdf.forw) * otherExt.neeConversion / vertex.ext().count;
					float misWeight = 1.0f / (1.0f + relSum + relHitPdf);
					radiance += (misWeight * bsdf.cosOut) * bsdf.value * otherExt.neeIrradiance;
				}
				++otherEndIt;
			}
		}

		// Evaluate direct hit of area ligths
		if(vertex.ext().pathLen >= m_params.minPathLength) {
			auto emission = vertex.get_emission();
			if(emission.value != 0.0f && vertex.ext().pathLen > 1) {
				AreaPdf startPdf = connect_pdf(m_sceneDesc.lightTree, vertex.get_primitive_id(),
											   vertex.get_surface_params(),
											   vertex.ext().previous->get_position(), guideFunction);
				float relPdf = startPdf / vertex.ext().incidentPdf;
				float relSum = relPdf + relPdf * vertex.ext().previous->ext().prevRelativeProbabilitySum;
				relSum *= vertex.ext().previous->ext().count;
				float misWeight = 1.0f / (1.0f + relSum);
				emission.value *= misWeight;
			}
			radiance += emission.value;
		}

		m_outputBuffer.contribute(vertex.ext().coord, { vertex.ext().throughput, 1.0f }, { Spectrum{1.0f}, 1.0f },
								  1.0f, radiance);
	}

	Profiler::instance().create_snapshot_all();
}

void CpuNextEventBacktracking::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_viewVertexMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
	m_viewVertexMap = m_viewVertexMapManager.acquire<Device::CPU>();
	m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1) * m_params.maxPathLength);
	m_photonMap = m_photonMapManager.acquire<Device::CPU>();
}

void CpuNextEventBacktracking::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer