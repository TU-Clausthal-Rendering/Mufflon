#pragma once

#include "ss_pt_params.hpp"
#include "ss_pt_common.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <ei/3dintersection.hpp>
#include <utility>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace ss {

using namespace scene::lights;

namespace {

inline float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

} // namespace

CUDA_FUNCTION void post_process_shadow(ss::SilhouetteTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
									   const scene::SceneDescriptor<CURRENT_DEV>& scene,
									   const SilhouetteParameters& params,
									   const Pixel& coord, const int pixel, const int iteration,
									   ArrayDevHandle_t<CURRENT_DEV, ShadowStatus> shadowStatus) {
	const auto lightCount = 1u + scene.lightTree.posLights.lightCount + scene.lightTree.dirLights.lightCount;

	ei::Vec2 statii{ 0.f };
	ei::Vec2 penumbra{ 0.f };
	for(int d = 0; d < params.maxPathLength; ++d) {
		for(std::size_t i = 0u; i < lightCount; ++i) {
			const auto status = shadowStatus[pixel * lightCount * params.maxPathLength + i * params.maxPathLength + d];

			// Detect if it is either penumbra or shadow edge

			if(status.shadow > 0.f) {
				const bool isPenumbra = status.light > 0.f;
				bool isEdge = false;
				for(int x = -1; x <= 1 && !isEdge; ++x) {
					for(int y = -1; y <= 1 && !isEdge; ++y) {
						if(x == y)
							continue;
						const auto currCoord = coord + Pixel{ x, y };
						if(currCoord.x < 0 || currCoord.y < 0 ||
						   currCoord.x >= outputBuffer.get_width() || currCoord.y >= outputBuffer.get_height())
							continue;
						const auto currPixel = currCoord.x + currCoord.y * outputBuffer.get_width();

						const auto currStatus = shadowStatus[currPixel * lightCount * params.maxPathLength + i * params.maxPathLength + d];
						if(currStatus.light > 0.f && currStatus.shadow == 0.f)
							isEdge = true;
					}
				}

				if(isPenumbra)
					penumbra.x += status.shadow;
				if(isEdge)
					penumbra.y += status.shadow;
			}
			statii += ei::Vec2{ status.shadow, status.light };
		}
	}
	outputBuffer.template set<ShadowTransitionTarget>(coord, statii / static_cast<float>(iteration + 1u));
	outputBuffer.template set<PenumbraTarget>(coord, penumbra / static_cast<float>(iteration + 1u));
}

CUDA_FUNCTION void sample_importance(ss::SilhouetteTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
									 const scene::SceneDescriptor<CURRENT_DEV>& scene,
									 const SilhouetteParameters& params,
									 const Pixel& coord, math::Rng& rng,
									 ShadowStatus* shadowStatus) {/*,
									 Importances<CURRENT_DEV>** importances,
									 DeviceImportanceSums<CURRENT_DEV>* sums,
									 SilhouetteEdge& shadowPrim, u8* penumbraBits) {*/
	Spectrum throughput{ ei::Vec3{1.0f} };
	SilPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	SilPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	int pathLen = 0;
	do {
		if(pathLen > 0 && pathLen + 1 >= params.minPathLength && pathLen + 1 <= params.maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			u64 neeSeed = rng.next();
			for(int i = 0; i < params.neeCount; ++i) {
				math::RndSet2 neeRnd = rng.next();

				u32 lightIndex;
				auto nee = scene::lights::connect(scene, i, params.neeCount,
												  neeSeed, vertex.get_position(), neeRnd,
												  &lightIndex);
				Pixel outCoord;
				auto value = vertex.evaluate(nee.dir.direction, scene.media, outCoord, nee.dist);
				if(nee.cosOut != 0) value.cosOut *= nee.cosOut;
				mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
				Spectrum radiance = value.value * nee.diffIrradiance;
				if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
					bool anyhit = scene::accel_struct::any_intersection(
						scene, vertex.get_position(), nee.position,
						vertex.get_geometric_normal(), nee.geoNormal,
						nee.dir.direction);

					AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (params.neeCount + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					const auto contribution = throughput * value.cosOut * radiance * mis;
					if(!anyhit) {
						outputBuffer.template contribute<RadianceTarget>(coord, contribution);
						shadowStatus[lightIndex * params.maxPathLength + pathLen].light += 1.f / (1.f + vertex.ext().footprint.get_solid_angle());
					} else {
						outputBuffer.template contribute<ShadowTarget>(coord, 1.f);
						shadowStatus[lightIndex * params.maxPathLength + pathLen].shadow += get_luminance(contribution) / (1.f + vertex.ext().footprint.get_solid_angle());
					}
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex.get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample, scene) == WalkResult::CANCEL)
			break;
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
		}
		if(vertex.is_end_point()) break;
	} while(pathLen < params.maxPathLength);
}

}}}}} // namespace mufflon::renderer::decimaters::silhouette::ss