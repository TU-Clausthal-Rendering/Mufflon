#pragma once

#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/renderer/output_handler.hpp"

namespace mufflon { namespace renderer {

struct PtVertexExt {
	//scene::Direction excident;
	//AngularPdf pdf;
	AreaPdf incidentPdf;

	CUDA_FUNCTION void init(const PathVertex<PtVertexExt>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const AreaPdf incidentPdf, const float incidentCosineAbs,
			  const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<PtVertexExt>& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair& pdf) {
		//excident = sample.excident;
		//pdf = sample.pdfF;
	}
};

using PtPathVertex = PathVertex<PtVertexExt>;


/*
 * Create one sample path (actual PT algorithm)
 */
CUDA_FUNCTION void pt_sample(RenderBuffer<CURRENT_DEV> outputBuffer,
							 const scene::SceneDescriptor<CURRENT_DEV>& scene,
							 const PtParameters& params,
							 const Pixel coord,
							 math::Rng& rng) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	//if(coord == Pixel{407, 499-437}) __debugbreak();

	int pathLen = 0;
	do {
		if(pathLen > 0 && pathLen+1 >= params.minPathLength && pathLen+1 <= params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			u64 neeSeed = rng.next();
			for(int i = 0; i < params.neeCount; ++i) {
				math::RndSet2 neeRnd = rng.next();
				auto nee = scene::lights::connect(scene, i, params.neeCount,
								   neeSeed, vertex.get_position(), neeRnd);
				Pixel outCoord;
				auto value = vertex.evaluate(nee.dir.direction, scene.media, outCoord);
				if(nee.cosOut != 0) value.cosOut *= nee.cosOut;
				mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
				Spectrum radiance = value.value * nee.diffIrradiance;
				if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
					bool anyhit = scene::accel_struct::any_intersection(
									scene, vertex.get_position(), nee.position,
									vertex.get_geometric_normal(), nee.geoNormal,
									nee.dir.direction);
					if(!anyhit) {
						AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
						float mis = 0;//1.f / params.neeCount;//1.0f / (params.neeCount + hitPdf / nee.creationPdf);
						mAssert(!isnan(mis));
						outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
												value.cosOut, radiance * mis);
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample) == WalkResult::CANCEL)
			break;
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen >= params.minPathLength) {
			EmissionValue emission = vertex.get_emission(scene, lastPosition);
			if(emission.value != 0.0f && pathLen > 1) {
				// misWeight for pathLen==1 is always 1 -> skip computation
				float misWeight = 1.0f;//1.0f / (1.0f + params.neeCount * (emission.connectPdf / vertex.ext().incidentPdf));
				emission.value *= misWeight;
			}
			outputBuffer.contribute(coord, throughput, emission.value, vertex.get_position(),
									vertex.get_normal(), vertex.get_albedo());
		}
		if(vertex.is_end_point()) break;
	} while(pathLen < params.maxPathLength);
}

}} // namespace mufflon::renderer