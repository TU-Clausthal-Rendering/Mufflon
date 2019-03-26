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


void sample_view_path(const scene::SceneDescriptor<CURRENT_DEV>& scene,
					  const NebParameters& params,
					  const Pixel coord,
					  math::Rng& rng,
					  HashGrid<Device::CPU, NebPathVertex>& viewVertexMap) {
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
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			u64 neeSeed = rng.next();
			for(int i = 0; i < params.neeCount; ++i) {
				math::RndSet2 neeRnd = rng.next();
				auto nee = connect(scene.lightTree, i, params.neeCount, neeSeed,
								   vertex.get_position(), scene.aabb, neeRnd,
								   guideFunction);
				Pixel outCoord;
				if(nee.cosOut != 0) nee.diffIrradiance *= nee.cosOut;
				if(any(greater(nee.diffIrradiance, 0.0f))) {
					bool anyhit = scene::accel_struct::any_intersection(
									scene, { vertex.get_position(), nee.direction },
									vertex.get_primitive_id(), nee.dist);
					if(!anyhit) {
						//AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
						vertex.ext().neeIrradiance = nee.diffIrradiance;
						vertex.ext().neeDirection = nee.direction;
					} else {
						vertex.ext().neeIrradiance = Spectrum{0.0f};
					}
				}
				vertex.ext().coord = coord;
				viewVertexMap.insert(vertex.get_position(), vertex);
			}
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
	m_viewVertexMap.clear(currentMergeRadius * 2.0001f);
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;

	// First pass: distribute and store view path vertices.
	// For each vertex compute the next event estimate, but do not contribute yet.
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		Pixel coord { pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() };
		sample_view_path(m_sceneDesc, m_params, coord, m_rngs[pixel], m_viewVertexMap);
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
		auto otherEndIt = m_viewVertexMap.find_first(vertex.get_position());
		while(otherEndIt) {
			int pathLen = 1;//otherEndIt->ext().pathLen + 1;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(otherEndIt->get_position() - currentPos) < mergeRadiusSq) {
				if(any(greater(otherEndIt->ext().neeIrradiance, 0.0f))) {
					Pixel tmpCoord;
					//scene::Direction geoNormal = otherEndIt->get_geometric_normal();
					auto bsdf = vertex.evaluate(otherEndIt->ext().neeDirection,
												m_sceneDesc.media, tmpCoord, false,
												nullptr);
					radiance += bsdf.cosOut * bsdf.value * otherEndIt->ext().neeIrradiance;
				}
				++count;
			}
			++otherEndIt;
		}
		if(count >= 1) radiance /= count;
		m_outputBuffer.contribute(vertex.ext().coord, vertex.ext().throughput, { Spectrum{1.0f}, 1.0f },
								  1.0f, radiance);
	}

	// Third pass: merge backtracked photons.

	Profiler::instance().create_snapshot_all();
}

void CpuNextEventBacktracking::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
	m_viewVertexMapManager.resize(m_outputBuffer.get_num_pixels() * (m_params.maxPathLength - 1));
	m_viewVertexMap = m_viewVertexMapManager.acquire<Device::CPU>();
}

void CpuNextEventBacktracking::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer