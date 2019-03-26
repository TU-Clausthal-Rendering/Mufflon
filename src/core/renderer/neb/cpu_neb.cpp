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
	//scene::Direction excident;
	//AngularPdf pdf;
	AreaPdf incidentPdf;
	math::Throughput throughput;
	Spectrum neeIrradiance;
	Pixel coord;
	scene::Direction neeDirection;
	int pathLen;

	CUDA_FUNCTION void init(const PathVertex<NebVertexExt>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const AreaPdf incidentPdf, const float incidentCosineAbs,
			  const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput;
	}

	CUDA_FUNCTION void update(const PathVertex<NebVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		//excident = sample.excident;
		//pdf = sample.pdfF;
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
	while(lightPathLength < params.maxPathLength-1
		&& walk(scene, virtualLight, rnd, rndRoulette, true, lightThroughput, virtualLight, sample)) {
		++lightPathLength;
		photonMap.insert(virtualLight.get_position(),
			{ virtualLight.get_position(), AreaPdf{0.0f},
				sample.excident, lightPathLength,
				lightThroughput.weight, virtualLight.get_geometric_normal()
			});
	}
}

void sample_view_path(const scene::SceneDescriptor<CURRENT_DEV>& scene,
					  const NebParameters& params,
					  const Pixel coord,
					  math::Rng& rng,
					  HashGrid<Device::CPU, NebPathVertex>& viewVertexMap,
					  HashGrid<Device::CPU, CpuNextEventBacktracking::PhotonDesc>& photonMap) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	NebPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	NebPathVertex::create_camera(&vertex, &vertex, scene.camera.get(), coord, rng.next());

	auto& guideFunction = params.neeUsePositionGuide ? scene::lights::guide_flux_pos
													 : scene::lights::guide_flux;

	int pathLen = 0;
	do {
		if(pathLen > 0 && pathLen+1 <= params.maxPathLength) {
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
				scene::Point hitPos = neePos - nee.direction * hit.hitT;
				const scene::TangentSpace tangentSpace = scene::accel_struct::tangent_space_geom_to_shader(scene, hit);
				NebPathVertex virtualLight;
				NebPathVertex::create_surface(&virtualLight, &virtualLight, hit,
					scene.get_material(hit.hitId), hitPos, tangentSpace, -nee.direction,
					nee.dist, AngularPdf{1.0f}, math::Throughput{});
				// Compensate the changed distance in diffIrradiance.
				virtualLight.ext().neeIrradiance = nee.diffIrradiance * nee.distSq / (hit.hitT * hit.hitT);
				virtualLight.ext().neeDirection = nee.direction;
				virtualLight.ext().coord = coord;
				virtualLight.ext().pathLen = -1;	// Mark this as non contributing (not connected to a pixel)
			//	viewVertexMap.insert(virtualLight.get_position(), virtualLight);
				// Make sure the vertex for which we did the NEE knows it is shadowed.
				nee.diffIrradiance *= 0.0f;
			}
			//}
			vertex.ext().neeIrradiance = nee.diffIrradiance;
			vertex.ext().neeDirection = nee.direction;
			vertex.ext().coord = coord;
			vertex.ext().pathLen = pathLen;
			viewVertexMap.insert(vertex.get_position(), vertex);
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(!walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
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

		// Evaluate direct hit of area ligths
		/*if(pathLen >= params.minPathLength) {
			Spectrum emission = vertex.get_emission().value;
			if(emission != 0.0f) {
				AreaPdf startPdf = connect_pdf(scene.lightTree, vertex.get_primitive_id(),
												  vertex.get_surface_params(),
												  lastPosition, guideFunction);
				float mis = pathLen == 1 ? 1.0f
					: 1.0f / (1.0f + params.neeCount * (startPdf / vertex.ext().incidentPdf));
				emission *= mis;
			}
			outputBuffer.contribute(coord, throughput, emission, vertex.get_position(),
									vertex.get_normal(), vertex.get_albedo());
		}*/
	} while(pathLen < params.maxPathLength);
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
	m_viewVertexMap.clear(currentMergeRadius * 2.0001f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);

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
		Spectrum radiance { 0.0f };
		int count = 0;
		auto otherEndIt = m_viewVertexMap.find_first(currentPos);
		while(otherEndIt) {
			int pathLen = vertex.ext().pathLen + 1;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(otherEndIt->get_position() - currentPos) < mergeRadiusSq) {
				if(any(greater(otherEndIt->ext().neeIrradiance, 0.0f))) {
					Pixel tmpCoord;
					auto bsdf = vertex.evaluate(otherEndIt->ext().neeDirection,
												m_sceneDesc.media, tmpCoord, false,
												nullptr);
					radiance += bsdf.cosOut * bsdf.value * otherEndIt->ext().neeIrradiance;
				}
				++count;
			}
			++otherEndIt;
		}
		mAssert(count >= 1);
		radiance /= count;
		//m_outputBuffer.contribute(vertex.ext().coord, vertex.ext().throughput, { Spectrum{1.0f}, 1.0f },
		//						  1.0f, radiance);

		NebPathVertex virtualLight = vertex;
		virtualLight.set_incident_direction(-vertex.ext().neeDirection);
		float cosLight = ei::abs(vertex.get_geometrical_factor(vertex.ext().neeDirection));
		virtualLight.ext().neeIrradiance *= cosLight / count;
		int rngIndex = i % m_outputBuffer.get_num_pixels();
		sample_photons(m_sceneDesc, m_params, m_rngs[rngIndex], virtualLight, m_photonMap);
	}

	// Third pass: merge backtracked photons.
#pragma PARALLEL_FOR
	for(i32 i = 0; i < numViewVertices; ++i) {
		auto& vertex = m_viewVertexMap.get_data_by_index(i);
		scene::Point currentPos = vertex.get_position();
		Spectrum radiance { 0.0f };
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			int pathLen = photonIt->pathLen + vertex.ext().pathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photonIt->position - currentPos) < mergeRadiusSq) {
				Pixel tmpCoord;
				auto bsdf = vertex.evaluate(-photonIt->incident,
											m_sceneDesc.media, tmpCoord, false,
											&photonIt->geoNormal);
				radiance += bsdf.value * photonIt->irradiance;
			}
			++photonIt;
		}
		m_outputBuffer.contribute(vertex.ext().coord, vertex.ext().throughput, { Spectrum{1.0f}, 1.0f },
								  1.0f, radiance);
	}

	Profiler::instance().create_snapshot_all();
}

void CpuNextEventBacktracking::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_viewVertexMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
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