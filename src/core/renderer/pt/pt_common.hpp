#pragma once

#include "pt_params.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/renderer/targets/render_targets.hpp"
#include "core/renderer/footprint.hpp"
#include <math.h>

namespace mufflon { namespace renderer {

inline CUDA_FUNCTION void update_guide_heuristic(float& guideWeight, int pathLen, AngularPdf pdfForw) {
	if(pathLen > 0) {
		float pSq = pdfForw * pdfForw;
		//guideWeight *= 1.0f - expf(-pSq / 5.0f);
		guideWeight *= pSq / (1000.0f + pSq);
		//const auto roughness = ei::sqrt(1.f / (ei::PI * static_cast<float>(pdfForw)));
		//guideWeight *= 0.5f * (1.f + std::tanh(-5.f * roughness + 3.f));
		//guideWeight *= 0.5f * (1.f + std::tanh(0.03f * static_cast<float>(pdfForw) - 3.f));
		guideWeight *= std::tanh(0.02f * static_cast<float>(pdfForw) + 0.1f);
	}
}

struct PtVertexExt {
	//scene::Direction excident;
	//AngularPdf pdf;
	AreaPdf incidentPdf;
	FootprintV0 footprint;

	inline CUDA_FUNCTION void init(const PathVertex<PtVertexExt>& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		const float sourceCount = 1.f;
		if(thisVertex.get_path_len() > 0)
			this->footprint.init(1.0f / (float(inAreaPdf) * sourceCount), 1.0f / (float(inDirPdf) * sourceCount), pChoice);
		else
			this->footprint.init(0.f, 0.f, pChoice);
	}

	inline CUDA_FUNCTION void update(const PathVertex<PtVertexExt>& prevVertex,
							  const PathVertex<PtVertexExt>& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const Spectrum& /*throughput*/,
							  const float /*continuationPropability*/,
							  const Spectrum& /*transmission*/,
							  float& guideWeight) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);

		this->footprint = prevVertex.ext().footprint.add_segment(
			static_cast<float>(pdf.forw), prevVertex.is_orthographic(),
			0.f, 0.f, 0.f, 1.f, incident.distance, 0.f, 1.0f);
		guideWeight = 1.f / (1.f + footprint.get_solid_angle());
	}

	inline CUDA_FUNCTION void update(const PathVertex<PtVertexExt>& /*thisVertex*/,
							  const scene::Direction& /*excident*/,
							  const VertexSample& /*sample*/,
							  float& /*guideWeight*/) {}
};

using PtPathVertex = PathVertex<PtVertexExt>;


/*
 * Create one sample path (actual PT algorithm)
 */
inline CUDA_FUNCTION void pt_sample(PtTargets::template RenderBufferType<CURRENT_DEV> outputBuffer,
							 const scene::SceneDescriptor<CURRENT_DEV>& scene,
							 const PtParameters& params,
							 const Pixel coord,
							 math::Rng& rng) {
	Spectrum throughput { 1.0f };
	float guideWeight = 1.0f;
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	//if(coord == Pixel{131, 540}) __debugbreak();

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
						float mis = 1.0f / (params.neeCount + hitPdf / nee.creationPdf);
						mAssert(!isnan(mis));
						outputBuffer.contribute<RadianceTarget>(coord, throughput * value.cosOut * radiance * mis);
						outputBuffer.contribute<LightnessTarget>(coord, guideWeight * value.cosOut);

					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd { rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) == WalkResult::CANCEL)
			break;

		if(pathLen == 0)
			outputBuffer.template contribute<HitIdTarget>(coord, ei::Vec2{ vertex.get_primitive_id().instanceId,
																		   vertex.get_primitive_id().primId });
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen >= params.minPathLength) {
			EmissionValue emission = vertex.get_emission(scene, lastPosition);
			if(emission.value != 0.0f && pathLen > 1) {
				// misWeight for pathLen==1 is always 1 -> skip computation
				float misWeight = 1.0f / (1.0f + params.neeCount * (emission.connectPdf / vertex.ext().incidentPdf));
				emission.value *= misWeight;
			}
			outputBuffer.template contribute<RadianceTarget>(coord, throughput * emission.value);
			outputBuffer.template contribute<PositionTarget>(coord, guideWeight * vertex.get_position());
			outputBuffer.template contribute<DepthTarget>(coord, guideWeight * vertex.get_incident_dist());
			outputBuffer.template contribute<NormalTarget>(coord, guideWeight * vertex.get_normal());
			outputBuffer.template contribute<AlbedoTarget>(coord, guideWeight * vertex.get_albedo());
			outputBuffer.template contribute<LightnessTarget>(coord, guideWeight * ei::avg(emission.value));
		}
		if(vertex.is_end_point()) break;
	} while(pathLen < params.maxPathLength);
}

}} // namespace mufflon::renderer
