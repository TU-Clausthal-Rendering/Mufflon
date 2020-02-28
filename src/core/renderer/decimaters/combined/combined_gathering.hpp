#pragma once

#include "combined_common.hpp"
#include "combined_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/renderer/decimaters/util/octree.inl"
#include "core/renderer/decimaters/util/float_octree.inl"
#include "core/scene/descriptors.hpp"
#include "core/scene/util.hpp"
#include "core/scene/lights/lights.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace renderer { namespace decimaters { namespace combined {

using namespace scene::lights;

template < class Octree >
inline CUDA_FUNCTION void distribute_sample(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon,
											const u32 primId, const ei::Vec3& objSpacePos,
											Octree& octree, const float value) {
	if(primId < polygon.numTriangles) {
		const auto tri = scene::get_triangle(polygon, primId);
		float distSum = 0.f;
		for(u32 i = 0u; i < 3u; ++i)
			distSum += ei::len(tri.v(i) - objSpacePos);
		for(u32 i = 0u; i < 3u; ++i) {
			const auto dist = ei::len(tri.v(i) - objSpacePos);
			// TODO: normal!
			octree.add_sample(tri.v(i), ei::Vec3{ 0.f, 1.f, 0.f }, value * dist / distSum);
		}
	} else {
		const auto quad = scene::get_quad(polygon, primId);
		float distSum = 0.f;
		for(u32 i = 0u; i < 4u; ++i)
			distSum += ei::len(quad.v(i) - objSpacePos);
		for(u32 i = 0u; i < 4u; ++i) {
			const auto dist = ei::len(quad.v(i) - objSpacePos);
			// TODO: normal!
			octree.add_sample(quad.v(i), ei::Vec3{ 0.f, 1.f, 0.f }, value * dist / distSum);
		}
	}
}

inline CUDA_FUNCTION void post_process_shadow(CombinedTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
											  const scene::SceneDescriptor<CURRENT_DEV>& scene,
											  const CombinedParameters& params,
											  const Pixel& coord, const int pixel, const int iterations,
											  ArrayDevHandle_t<CURRENT_DEV, ShadowStatus> shadowStatus) {
	const auto lightCount = 1u + scene.lightTree.posLights.lightCount + scene.lightTree.dirLights.lightCount;

	ei::Vec2 statii{ 0.f };
	ei::Vec3 penumbra{ 0.f };
	for(int d = 0; d < params.maxPathLength - 1; ++d) {
		for(std::size_t i = 0u; i < lightCount; ++i) {
			const auto status = shadowStatus[pixel * lightCount * (params.maxPathLength - 1) + i * (params.maxPathLength - 1) + d];

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

						const auto currStatus = shadowStatus[currPixel * lightCount * (params.maxPathLength - 1)
							+ i * (params.maxPathLength - 1) + d];
						if(currStatus.light > 0.f && currStatus.shadow == 0.f)
							isEdge = true;
					}
				}

				if(isPenumbra) {
					const auto radiance = scene::get_luminance(outputBuffer.template get<RadianceTarget>(coord));
					penumbra.x += status.shadow / static_cast<float>(iterations);
					penumbra.z += 1.f + status.shadowContributions / (radiance * static_cast<float>(iterations));
					// TODO!
					//penumbra.z = (scene::get_luminance(outputBuffer.template get<RadianceTarget>(coord))
					//	- status.shadowContributions) / static_cast<float>(iterations);
				}
				if(isEdge)
					penumbra.y += status.shadow / static_cast<float>(iterations);
			}
			statii += ei::Vec2{ status.shadow, status.light };
		}
	}
	outputBuffer.template set<PenumbraTarget>(coord, penumbra);
}

inline CUDA_FUNCTION void sample_importance_octree(CombinedTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
												   const scene::SceneDescriptor<CURRENT_DEV>& scene,
												   const CombinedParameters& params,
												   const Pixel& coord, math::Rng& rng,
												   FloatOctree* viewOctrees,
												   SampleOctree* irradianceOctrees,
												   ShadowStatus* shadowStatus) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	// We gotta keep track of our vertices
	// TODO: flexible length!
#ifdef __CUDA_ARCH__
	CombinedPathVertex vertices[16u];
#else // __CUDA_ARCH__
	static thread_local CombinedPathVertex vertices[16u];
#endif // __CUDA_ARCH__
	if(params.maxPathLength >= sizeof(vertices) / sizeof(*vertices))
		mAssertMsg(false, "Path length out of bounds!");
	//thread_local std::vector<CombinedPathVertex> vertices(std::max(2, params.maxPathLength + 1));
	//vertices.clear();
	// Create a start for the path
	(void)CombinedPathVertex::create_camera(&vertices[0], &vertices[0], scene.camera.get(), coord, rng.next());
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
			auto value = vertices[pathLen].evaluate(nee.dir.direction, scene.media, projCoord, nee.dist);
			if(nee.cosOut != 0) value.cosOut *= nee.cosOut;
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			const Spectrum radiance = value.value * nee.diffIrradiance;
			// TODO: use multiple NEEs
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				ei::Ray shadowRay{ nee.position, -nee.dir.direction };

				// TODO: any_intersection-like method with both normals please...
				const auto shadowHit = scene::accel_struct::first_intersection(scene, shadowRay,
																			   nee.geoNormal,
																			   nee.dist - 0.0125f);
				const float firstShadowDistance = shadowHit.distance;
				const AreaPdf hitPdf = value.pdf.forw.to_area_pdf(nee.cosOut, nee.distSq);
				const float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
				const ei::Vec3 irradiance = nee.diffIrradiance * value.cosOut; // [W/m²]
				vertices[pathLen].ext().pathRadiance = mis * radiance * value.cosOut;

				const float weightedRadianceLuminance = scene::get_luminance(throughput * mis * radiance)
					* (1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				const float weightedIrradianceLuminance = scene::get_luminance(throughput * irradiance)
					*(1.f - ei::abs(vertices[pathLen - 1].ext().outCos));
				const auto contribution = throughput * value.cosOut * radiance * mis;
				if(shadowHit.hitId.instanceId < 0) {
					outputBuffer.template contribute<RadianceTarget>(coord, contribution);
					shadowStatus[lightIndex * (params.maxPathLength - 1) + pathLen].light += scene::get_luminance(contribution) /
						(1.f + vertices[pathLen].ext().footprint.get_solid_angle());
					if(params.show_direct()) {
						mAssert(!isnan(mis));
						// TODO
						const auto hitId = vertices[pathLen].get_primitive_id();
						const auto objSpacePos = ei::transform(vertices[pathLen].get_position(),
															   scene.worldToInstance[hitId.instanceId]);
						const auto objSpaceNormal = ei::transform(vertices[pathLen].get_normal(),
																  ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
						// Determine the face area
						const auto lodIdx = scene.lodIndices[hitId.instanceId];
						const auto area = compute_area_instance_transformed(scene, scene.lods[lodIdx].polygon, hitId);
						const auto baseArea = compute_area(scene, scene.lods[lodIdx].polygon, hitId);
						distribute_sample(scene.lods[lodIdx].polygon, static_cast<u32>(hitId.primId),
										  objSpacePos, irradianceOctrees[lodIdx], params.lightWeight * baseArea / area);
					}
				} else {
					shadowStatus[lightIndex * (params.maxPathLength - 1) + pathLen].shadow += scene::get_luminance(contribution) /
						(1.f + vertices[pathLen].ext().footprint.get_solid_angle());
					shadowStatus[lightIndex * (params.maxPathLength - 1) + pathLen].shadowContributions += scene::get_luminance(contribution);

					if(pathLen == 1) {
						// Determine the "rest of the direct" radiance
						const u64 ambientNeeSeed = rng.next();
						ei::Vec3 rad{ 0.f };
						const int neeCount = ei::max<int>(1, params.neeCount);
						const scene::Point vertexPos = vertices[pathLen].get_position();
						const scene::Point vertexNormal = vertices[pathLen].get_geometric_normal();
						for(int i = 0; i < neeCount; ++i) {
							math::RndSet2 currNeeRnd = rng.next();
							auto currNee = scene::lights::connect(scene, i, neeCount,
																  ambientNeeSeed, vertexPos,
																  currNeeRnd);
							Pixel outCoord;
							auto currValue = vertices[pathLen].evaluate(currNee.dir.direction, scene.media, outCoord);
							if(currNee.cosOut != 0) value.cosOut *= currNee.cosOut;
							mAssert(!isnan(currValue.value.x) && !isnan(currValue.value.y) && !isnan(currValue.value.z));
							const Spectrum currRadiance = currValue.value * currNee.diffIrradiance;
							if(any(greater(currRadiance, 0.0f)) && currValue.cosOut > 0.0f) {
								bool anyhit = scene::accel_struct::any_intersection(
									scene, vertexPos, currNee.position,
									vertexNormal, currNee.geoNormal,
									currNee.dir.direction);
								if(!anyhit) {
									AreaPdf currHitPdf = currValue.pdf.forw.to_area_pdf(currNee.cosOut, currNee.distSq);
									// TODO: it seems that, since we're looking at irradiance here (and also did not weight
									// the previous weightedIrradiance with 1/(neeCount + 1)) we must not use the regular
									// MIS weight here
									float curMis = 1.0f / (1 + currHitPdf / currNee.creationPdf);
									mAssert(!isnan(curMis));
									rad += currValue.cosOut * currRadiance * curMis;
								}
							}
						}

						vertices[pathLen].ext().otherNeeLuminance = scene::get_luminance(rad);

						// TODO
					}
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
			const float bxdfLum = scene::get_luminance(bxdf);
			if(isnan(bxdfLum))
				return;
			// TODO!
			sharpness *= 2.f / (1.f + ei::exp(-bxdfLum / 10.f)) - 1.f;
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
			const auto baseArea = compute_area(scene, lod.polygon, hitId);

			const auto lodIdx = scene.lodIndices[hitId.instanceId];

			const auto impDensity = sharpness * (1.f - ei::abs(cosAngle)) * baseArea / area;
			if(!isnan(impDensity))
				distribute_sample(scene.lods[lodIdx].polygon, static_cast<u32>(hitId.primId),
								  objSpacePos, viewOctrees[lodIdx], impDensity);
		}

		vertices[pathLen].ext().pathRadiance = ei::Vec3{ 0.f };
		if(pathLen >= 0) {
			EmissionValue emission = vertices[pathLen].get_emission(scene, lastPosition);
			if(emission.value != 0.0f && pathLen > 1) {
				// misWeight for pathLen==1 is always 1 -> skip computation
				float misWeight = 1.0f / (1.0f + params.neeCount * (emission.connectPdf / vertices[pathLen].ext().incidentPdf));
				emission.value *= misWeight;
			}
			outputBuffer.template contribute<RadianceTarget>(coord, emission.value * throughput);
			vertices[pathLen].ext().pathRadiance = emission.value * throughput;
		}
		if(vertices[pathLen].is_end_point()) break;
	} while(pathLen < params.maxPathLength);

	
	// TODO
	// Go back over the path and add up the irradiance from indirect illumination
	ei::Vec3 accumRadiance{ 0.f };
	ei::Vec3 currThroughput = throughput;
	for(int p = pathLen; p >= 1; --p) {
		// Last vertex doesn't have indirect contribution
		if(p < pathLen) {
			// Compute the throughput at the target vertex by recursively removing later throughputs
			currThroughput /= vertices[p].ext().throughput;
			accumRadiance = currThroughput * (accumRadiance + vertices[p + 1].ext().pathRadiance);
			const ei::Vec3 irradiance = ei::abs(vertices[p].ext().outCos) * accumRadiance;

			const auto& hitId = vertices[p].get_primitive_id();
			const auto* lod = &scene.lods[scene.lodIndices[hitId.instanceId]];
			const u32 numVertices = hitId.primId < (i32)lod->polygon.numTriangles ? 3u : 4u;

			const float importance = scene::get_luminance(irradiance) * (1.f - ei::abs(vertices[p].ext().outCos));
			if(params.show_indirect()) {
				const auto objSpacePos = ei::transform(vertices[p].get_position(),
													   scene.worldToInstance[hitId.instanceId]);
				const auto objSpaceNormal = ei::transform(vertices[p].get_normal(),
														  ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
				// Determine the face area
				const auto lodIdx = scene.lodIndices[hitId.instanceId];
				const auto area = compute_area_instance_transformed(scene, scene.lods[lodIdx].polygon, hitId);
				const auto baseArea = compute_area(scene, scene.lods[lodIdx].polygon, hitId);

				distribute_sample(scene.lods[lodIdx].polygon, static_cast<u32>(hitId.primId),
								  objSpacePos, irradianceOctrees[lodIdx], params.lightWeight * importance * baseArea / area);
			}
		}

	}
}


inline CUDA_FUNCTION void sample_vis_importance_octree(CombinedTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
													   const scene::SceneDescriptor<CURRENT_DEV>& scene,
													   const Pixel& coord, math::Rng& rng,
													   const FloatOctree* view) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	float guideWeight = 1.0f;
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) == WalkResult::HIT) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		if(hitId.primId >= scene.lods[lodIdx].numPrimitives)
			return;
		const auto polygon = scene.lods[lodIdx].polygon;

		const auto objSpacePos = ei::transform(vertex.get_position(),
											   scene.worldToInstance[hitId.instanceId]);

		// Iterate over the vertices and interpolate
		float distSum = 0.f;
		float viewImp = 0.f;
		if(static_cast<u32>(hitId.primId) < polygon.numTriangles) {
			const auto tri = scene::get_triangle(polygon, hitId.primId);
			for(u32 i = 0u; i < 3u; ++i) {
				const auto dist = ei::len(tri.v(i) - objSpacePos);
				viewImp += dist * view[lodIdx].get_samples(tri.v(i));
				distSum += dist;
			}
		} else {
			const auto quad = scene::get_quad(polygon, hitId.primId);
			for(u32 i = 0u; i < 4u; ++i) {
				const auto dist = ei::len(quad.v(i) - objSpacePos);
				viewImp += dist * view[lodIdx].get_samples(quad.v(i));
				distSum += dist;
			}
		}
		const auto area = scene::compute_area(scene, polygon, hitId);

		const auto importance = viewImp / (area * distSum);
		outputBuffer.template set<silhouette::ImportanceTarget>(coord, importance);
	}
}

inline CUDA_FUNCTION void sample_vis_importance(CombinedTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
												const scene::SceneDescriptor<CURRENT_DEV>& scene,
												const Pixel& coord, math::Rng& rng,
												const ConstArrayDevHandle_t<CURRENT_DEV, ConstArrayDevHandle_t<CURRENT_DEV, float>> importances) {
	Spectrum throughput{ ei::Vec3{1.0f} };
	float guideWeight = 1.0f;
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample, guideWeight) == WalkResult::HIT) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		if(hitId.primId >= scene.lods[lodIdx].numPrimitives)
			return;

		const auto polygon = scene.lods[lodIdx].polygon;
		const auto objSpacePos = ei::transform(vertex.get_position(),
											   scene.worldToInstance[hitId.instanceId]);

		float imp = 0.f;
		float distSum = 0.f;
		if(static_cast<u32>(hitId.primId) < polygon.numTriangles) {
			const auto indices = scene::get_triangle_vertex_indices(polygon, hitId.primId);
			const auto tri = scene::get_triangle(polygon, hitId.primId);
			for(u32 i = 0u; i < 3u; ++i) {
				const auto dist = ei::len(tri.v(i) - objSpacePos);
				imp += dist * importances[lodIdx][indices[i]];
				distSum += dist;
			}
		} else {
			const auto indices = scene::get_quad_vertex_indices(polygon, hitId.primId);
			const auto quad = scene::get_triangle(polygon, hitId.primId);
			for(u32 i = 0u; i < 4u; ++i) {
				const auto dist = ei::len(quad.v(i) - objSpacePos);
				imp += dist * importances[lodIdx][indices[i]];
				distSum += dist;
			}
		}
		const auto importance = imp / distSum;
		outputBuffer.template set<silhouette::ImportanceTarget>(coord, importance);
	}
}

}}}} // namespace mufflon::renderer::decimaters::combined
