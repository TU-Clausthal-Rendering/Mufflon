#pragma once

#include "silhouette_pt_common.hpp"
#include "silhouette_pt_params.hpp"
#include "core/data_structs/dm_hashgrid.hpp"
#include "core/data_structs/count_octree.hpp"
#include "core/export/core_api.h"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/util.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <ei/3dintersection.hpp>
#include <utility>

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace pt {

using namespace scene::lights;

inline CUDA_FUNCTION void sample_importance_octree(pt::SilhouetteTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
												   const scene::SceneDescriptor<CURRENT_DEV>& scene,
												   const SilhouetteParameters& params,
												   const Pixel& coord, math::Rng& rng,
												   DeviceImportanceSums<CURRENT_DEV>* sums,
												   data_structs::CountOctreeManager& viewOctrees,
												   data_structs::CountOctreeManager& irradianceOctrees) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	// We gotta keep track of our vertices
	// TODO: flexible length!
#ifdef __CUDA_ARCH__
	SilPathVertex vertices[16u];
#else // __CUDA_ARCH__
	static thread_local SilPathVertex vertices[16u];
#endif // __CUDA_ARCH__
	if(params.maxPathLength >= sizeof(vertices) / sizeof(*vertices))
		mAssertMsg(false, "Path length out of bounds!");
	//thread_local std::vector<SilPathVertex> vertices(std::max(2, params.maxPathLength + 1));
	//vertices.clear();
	// Create a start for the path
	(void)SilPathVertex::create_camera(&vertices[0], &vertices[0], scene.camera.get(), coord, rng.next());
	vertices[0].ext().pathRadiance = ei::Vec3{ 0.f };

	float sharpness = 1.f;

	// Andreas' algorithm mapped to path tracing:
	// Increasing importance for photons is equivalent to increasing
	// importance by the irradiance. Thus we need to compute "both
	// direct and indirect" irradiance for the path tracer as well.
	// They differ, however, in the types of paths that they
	// preferably find.

	int pathLen = 0;
	do {
		// Add direct contribution as importance as well
		if(pathLen > 0 && pathLen + 1 <= params.maxPathLength) {
			u64 neeSeed = rng.next();
			math::RndSet2 neeRnd = rng.next();
			u32 lightIndex;
			u32 lightOffset;
			scene::lights::LightType lightType;
			auto nee = scene::lights::connect(scene, 0, 1, neeSeed, vertices[pathLen].get_position(), neeRnd,
											  &lightIndex, &lightType, &lightOffset);
			Pixel projCoord;
			auto value = vertices[pathLen].evaluate(nee.dir.direction, scene.media, projCoord);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			// TODO: use multiple NEEs
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				ei::Ray shadowRay{ nee.position, -nee.dir.direction };

				// TODO: any_intersection-like method with both normals please...
				const auto shadowHit = scene::accel_struct::first_intersection(scene, shadowRay,
																			   nee.geoNormal,
																			   nee.dist - 0.000125f);
				const float firstShadowDistance = shadowHit.distance;
				AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

				const float weightedRadianceLuminance = get_luminance(throughput * mis * radiance) * (1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				const float weightedIrradianceLuminance = get_luminance(throughput * irradiance) *(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				if(shadowHit.hitId.instanceId < 0) {
					if(params.show_direct()) {
						mAssert(!isnan(mis));

						// TODO
					}
				} else if(pathLen == 1) {
					// TODO
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertices[pathLen].get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		VertexSample sample;
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertices[pathLen], rnd, rndRoulette, false, throughput, vertices[pathLen + 1], sample) != WalkResult::HIT)
			break;

		// Update old vertex with accumulated throughput
		vertices[pathLen].ext().updateBxdf(sample, throughput);

		// Don't update sharpness for camera vertex
		if(pathLen > 0) {
			// TODO: this seems to give the wrong result for dirac-delta BxDFs
			const ei::Vec3 bxdf = sample.throughput * (float)sample.pdf.forw;
			const float bxdfLum = get_luminance(bxdf);
			if(isnan(bxdfLum))
				return;
			sharpness *= 2.f / (1.f + ei::exp(-bxdfLum / params.sharpnessFactor)) - 1.f;
		}

		// Fetch the relevant information for attributing the instance to the correct vertices
		const auto& hitId = vertices[pathLen + 1].get_primitive_id();
		const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
		const u32 numVertices = hitId.primId < (i32)lod.polygon.numTriangles ? 3u : 4u;

		++pathLen;

		// Don't record direct hit importance for last path segment - that is only for NEE MIS
		if(params.show_view() && (pathLen < params.maxPathLength)) {
			const auto cosAngle = -ei::dot(vertices[pathLen].get_incident_direction(),
										   vertices[pathLen].get_normal());
			const auto objSpacePos = ei::transform(vertices[pathLen].get_position(),
												   scene.worldToInstance[hitId.instanceId]);
			const auto objSpaceNormal = ei::transform(vertices[pathLen].get_normal(),
													  ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
			// Determine the face area
			const auto area = compute_area_instance_transformed(scene, lod.polygon, hitId);

			const auto lodIdx = scene.lodIndices[hitId.instanceId];

			const auto impDensity = sharpness * (1.f - ei::abs(cosAngle)) / area;
			if(!isnan(impDensity))
				viewOctrees[lodIdx].add_sample(objSpacePos, objSpaceNormal, impDensity);
		}

		vertices[pathLen].ext().pathRadiance = ei::Vec3{ 0.f };
		if(pathLen >= params.minPathLength) {
			EmissionValue emission = vertices[pathLen].get_emission(scene, lastPosition);
			if(emission.value != 0.0f && pathLen > 1) {
				// misWeight for pathLen==1 is always 1 -> skip computation
				float misWeight = 1.0f / (1.0f + params.neeCount * (emission.connectPdf / vertices[pathLen].ext().incidentPdf));
				emission.value *= misWeight;
			}
			vertices[pathLen].ext().pathRadiance = emission.value * throughput;
		}
		if(vertices[pathLen].is_end_point()) break;
	} while(pathLen < params.maxPathLength);

	// TODO
}

}}}}} // namespace mufflon::renderer::decimaters::silhouette::pt
