#pragma once

#include "silhouette_bpm_common.hpp"
#include "silhouette_bpm_params.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/pt/pt_common.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

namespace mufflon { namespace renderer { namespace decimaters { namespace silhouette { namespace bpm {

namespace {

CUDA_FUNCTION constexpr float get_luminance(const ei::Vec3& vec) {
	constexpr ei::Vec3 LUM_WEIGHT{ 0.212671f, 0.715160f, 0.072169f };
	return ei::dot(LUM_WEIGHT, vec);
}

CUDA_FUNCTION void record_direct_hit(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
									 const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint,
									 const float cosAngle, const float sharpness) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(cosAngle))
		cuda::atomic_add<CURRENT_DEV>(importances[min].fluxImportance, sharpness * (1.f - ei::abs(cosAngle)));
}

CUDA_FUNCTION void record_flux(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
							   const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint, const float flux) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(flux)) {
		cuda::atomic_add<CURRENT_DEV>(importances[min].fluxImportance, flux);
	}
}

#if 0
CUDA_FUNCTION void record_silhouette_vertex_contribution(Importances<CURRENT_DEV>* importances,
														 DeviceImportanceSums<CURRENT_DEV>& sums,
														 const u32 vertexIndex, const float importance) {
	// Reminder: local index will refer to the decimated mesh
	cuda::atomic_add<CURRENT_DEV>(importances[vertexIndex].viewImportance, importance);
	cuda::atomic_add<CURRENT_DEV>(sums.shadowSilhouetteImportance, importance);
}

CUDA_FUNCTION void record_shadow(DeviceImportanceSums<CURRENT_DEV>& sums, const float irradiance) {
	cuda::atomic_add<CURRENT_DEV>(sums.shadowImportance, irradiance);
}

CUDA_FUNCTION void record_indirect_irradiance(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon, Importances<CURRENT_DEV>* importances,
											  const u32 primId, const u32 vertexCount, const ei::Vec3& hitpoint, const float irradiance) {
	const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
	const u32 primIdx = vertexCount == 3u ? primId : (primId - polygon.numTriangles);

	i32 min;
	float minDist = std::numeric_limits<float>::max();
	for(u32 v = 0u; v < vertexCount; ++v) {
		i32 vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + v];
		const float dist = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
		if(dist < minDist) {
			minDist = dist;
			min = vertexIndex;
		}
	}

	if(!isnan(irradiance))
		cuda::atomic_add<CURRENT_DEV>(importances[min].irradiance, irradiance);
}

CUDA_FUNCTION bool trace_shadow_silhouette(const scene::SceneDescriptor<CURRENT_DEV>& scene, Importances<CURRENT_DEV>** importances,
										   DeviceImportanceSums<CURRENT_DEV>* sums, const ei::Ray& shadowRay,
										   const SilPathVertex& vertex, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(scene, backfaceRay, vertex.get_geometric_normal(),
																   vertex.ext().lightDistance - vertex.ext().firstShadowDistance + DIST_EPSILON);
	// We absolutely have to have a second hit - either us (since we weren't first hit) or something else
	if(secondHit.hitId.instanceId >= 0 && secondHit.hitId != vertex.get_primitive_id()
	   && secondHit.hitId.instanceId == vertex.ext().shadowHit.instanceId) {
		// Check for silhouette - get the vertex indices of the primitives
		const auto& obj = scene.lods[scene.lodIndices[vertex.ext().shadowHit.instanceId]];
		const i32 firstNumVertices = vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 secondNumVertices = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 3 : 4;
		const i32 firstPrimIndex = vertex.ext().shadowHit.primId - (vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles
																	? 0 : (i32)obj.polygon.numTriangles);
		const i32 secondPrimIndex = secondHit.hitId.primId - (secondHit.hitId.primId < (i32)obj.polygon.numTriangles
															  ? 0 : (i32)obj.polygon.numTriangles);
		const i32 firstVertOffset = vertex.ext().shadowHit.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;
		const i32 secondVertOffset = secondHit.hitId.primId < (i32)obj.polygon.numTriangles ? 0 : 3 * (i32)obj.polygon.numTriangles;

		// Check if we have "shared" vertices: cannot do it by index, since they might be
		// split vertices, but instead need to go by proximity
		i32 sharedVertices = 0;
		i32 edgeIdxFirst[2];
		i32 edgeIdxSecond[2];
		for(i32 i0 = 0; i0 < firstNumVertices; ++i0) {
			for(i32 i1 = 0; i1 < secondNumVertices; ++i1) {
				const i32 idx0 = obj.polygon.vertexIndices[firstVertOffset + firstNumVertices * firstPrimIndex + i0];
				const i32 idx1 = obj.polygon.vertexIndices[secondVertOffset + secondNumVertices * secondPrimIndex + i1];
				const ei::Vec3& p0 = obj.polygon.vertices[idx0];
				const ei::Vec3& p1 = obj.polygon.vertices[idx1];
				if(idx0 == idx1 || p0 == p1) {
					edgeIdxFirst[sharedVertices] = idx0;
					edgeIdxSecond[sharedVertices] = idx1;
					++sharedVertices;
				}
				if(sharedVertices >= 2)
					break;
			}
		}

		if(sharedVertices >= 1) {
			// Got at least a silhouette point - now make sure we're seeing the silhouette
			ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + secondHit.distance), shadowRay.direction };

			/*const auto thirdHit = scene::accel_struct::any_intersection(m_sceneDesc, silhouetteRay, secondHit.hitId,
																		vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.distance + DIST_EPSILON);
			if(!thirdHit) {*/
			const auto thirdHit = scene::accel_struct::first_intersection(scene, silhouetteRay, secondHit.normal,
																		  vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.distance + DIST_EPSILON);
			if(thirdHit.hitId == vertex.get_primitive_id()) {
				for(i32 i = 0; i < sharedVertices; ++i) {
					const auto lodIdx = scene.lodIndices[vertex.ext().shadowHit.instanceId];
					record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx], edgeIdxFirst[i], importance);
					record_silhouette_vertex_contribution(importances[lodIdx], sums[lodIdx], edgeIdxSecond[i], importance);
				}
				return true;
			} else {
				mAssert(thirdHit.hitId.instanceId >= 0);
				// TODO: store a shadow photon?
			}
		}
	}
	return false;
}

CUDA_FUNCTION bool trace_shadow(const scene::SceneDescriptor<CURRENT_DEV>& scene, DeviceImportanceSums<CURRENT_DEV>* sums,
								const ei::Ray& shadowRay, const SilPathVertex& vertex, const float importance) {
	constexpr float DIST_EPSILON = 0.001f;

	// TODO: worry about spheres?
	ei::Ray backfaceRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + DIST_EPSILON), shadowRay.direction };

	const auto secondHit = scene::accel_struct::first_intersection(scene, backfaceRay, vertex.get_geometric_normal(),
																   vertex.ext().lightDistance - vertex.ext().firstShadowDistance + DIST_EPSILON);
	if(secondHit.hitId.instanceId < 0 || secondHit.hitId == vertex.get_primitive_id())
		return false;

	ei::Ray silhouetteRay{ shadowRay.origin + shadowRay.direction * (vertex.ext().firstShadowDistance + secondHit.distance), shadowRay.direction };
	const auto thirdHit = scene::accel_struct::first_intersection(scene, silhouetteRay, secondHit.normal,
																  vertex.ext().lightDistance - vertex.ext().firstShadowDistance - secondHit.distance + DIST_EPSILON);
	if(thirdHit.hitId == vertex.get_primitive_id()) {
		const auto& hitId = vertex.ext().shadowHit;
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		record_shadow(sums[lodIdx], importance);
		return true;
	}
	return false;
}
#endif // 0

// MIS weight for merges
CUDA_FUNCTION float get_mis_weight(const SilPathVertex& viewVertex, const math::PdfPair pdf,
								   const PhotonDesc& photon) {
	// Add the merge at the previous view path vertex
	mAssert(viewVertex.previous() != nullptr);
	float relPdf = viewVertex.ext().prevConversionFactor * float(pdf.back)
		/ float(viewVertex.ext().incidentPdf);
	float otherProbSum = relPdf + relPdf * viewVertex.previous()->ext().prevRelativeProbabilitySum;
	// Add the merge or hit at the previous light path vertex
	AreaPdf nextLightPdf{ float(pdf.forw) * photon.prevConversionFactor };
	relPdf = nextLightPdf / photon.incidentPdf;
	otherProbSum += relPdf + relPdf * photon.prevPrevRelativeProbabilitySum;
	return 1.0f / (1.0f + otherProbSum);
}

// MIS weight for unidirectional hits.
CUDA_FUNCTION float get_mis_weight(const SilPathVertex& thisVertex, const AngularPdf pdfBack,
								   const AreaPdf startPdf, int numPhotons, float mergeArea) {
	mAssert(thisVertex.previous() != nullptr);
	// There is one merge which is not yet accounted for
	float relPdf = thisVertex.ext().prevConversionFactor * float(pdfBack)
		/ float(thisVertex.ext().incidentPdf);
	// Until now, merges where only compared to other merges. The reuse factor and the
	// merge area are not yet included.
	relPdf *= float(startPdf) * numPhotons * mergeArea;
	float mergeProbSum = relPdf + relPdf * thisVertex.previous()->ext().prevRelativeProbabilitySum;
	return 1.0f / (1.0f + mergeProbSum);
}

} // namespace

CUDA_FUNCTION void trace_importance_photon(const scene::SceneDescriptor<CURRENT_DEV>& scene,
										   HashGrid<CURRENT_DEV, PhotonDesc>& photonMap,
										   const SilhouetteParameters& params,
										   const int idx, const int photonCount,
										   const u64 photonSeed, const float mergeRadius,
										   math::Rng& rng) {
	math::RndSet2_1 rndStart{ rng.next(), rng.next() };
	scene::lights::Emitter p = scene::lights::emit(scene, idx, photonCount, photonSeed, rndStart);
	SilPathVertex vertex[2];
	SilPathVertex::create_light(&vertex[0], nullptr, p, rng);	// TODO: check why there is an (unused) Rng reference
	math::Throughput throughput;
	float mergeArea = ei::PI * mergeRadius * mergeRadius;

	int pathLen = 0;
	int currentV = 0;
	int otherV = 1;
	PhotonDesc* prevPhoton = nullptr;
	do {
		// Walk
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		VertexSample sample;
		if(walk(scene, vertex[currentV], rnd, rndRoulette, true, throughput, vertex[otherV], sample) != WalkResult::HIT)
			break;
		++pathLen;
		currentV = otherV;
		otherV = 1 - currentV;
		// Complete the convertion factor with the quantities which where not known
		// to ext().init().
		if(pathLen == 1)
			vertex[currentV].ext().prevConversionFactor /= photonCount * mergeArea;

		const float angle = -ei::dot(vertex[currentV].get_normal(), vertex[currentV].get_incident_direction());
		// Store a photon to the photon map
		prevPhoton = photonMap.insert(vertex[currentV].get_position(),
									  { vertex[currentV].get_position(), vertex[currentV].ext().incidentPdf,
									  vertex[currentV].get_incident_direction(), pathLen,
									  angle * throughput.weight / photonCount, vertex[otherV].ext().prevRelativeProbabilitySum,
									  vertex[currentV].get_geometric_normal(), vertex[currentV].ext().prevConversionFactor,
									  prevPhoton, vertex[currentV].get_primitive_id() });

	} while(pathLen < params.maxPathLength - 1); // -1 because there is at least one segment on the view path
}


CUDA_FUNCTION void sample_importance(const scene::SceneDescriptor<CURRENT_DEV>& scene,
									 const HashGrid<CURRENT_DEV, PhotonDesc>& photonMap,
									 const SilhouetteParameters& params,
									 const Pixel& coord, math::Rng& rng,
									 const int photonCount, const float mergeRadius,
									 Importances<CURRENT_DEV>** importances,
									 DeviceImportanceSums<CURRENT_DEV>* sums) {
	float mergeRadiusSq = mergeRadius * mergeRadius;
	float mergeAreaInv = 1.0f / (ei::PI * mergeRadiusSq);
	// Trace view path
	SilPathVertex vertex[2];
	// Create a start for the path
	SilPathVertex::create_camera(&vertex[0], nullptr, scene.camera.get(), coord, rng.next());
	math::Throughput throughput;
	int currentV = 0;
	int viewPathLen = 0;
	do {
		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		VertexSample sample;
		if(walk(scene, vertex[currentV], rnd, rndRoulette, false, throughput, vertex[otherV], sample) == WalkResult::CANCEL)
			break;
		++viewPathLen;
		currentV = otherV;

		// TODO: shadow ray
		if(vertex[currentV].is_end_point()) break;

		{	// View importance
			const auto& hitId = vertex[currentV].get_primitive_id();
			const auto lodIdx = scene.lodIndices[hitId.instanceId];
			const auto& polygon = scene.lods[lodIdx].polygon;
			const u32 numVertices = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
			// TODO: sharpness
			record_direct_hit(polygon, importances[lodIdx], hitId.primId, numVertices, vertex[currentV].get_position(),
							  -ei::dot(vertex[currentV].get_incident_direction(), vertex[currentV].get_normal()),
							  params.viewWeight * 1.f);
		}

		// Merges
		scene::Point currentPos = vertex[currentV].get_position();
		auto photonIt = photonMap.find_first(currentPos);
		while(photonIt) {
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int pathLen = viewPathLen + photonIt->pathLen;
			if(pathLen >= params.minPathLength && pathLen <= params.maxPathLength
			   && lensq(photonIt->position - currentPos) < mergeRadiusSq) {
				// Importance attribution
				const auto& hitId = photonIt->hitId;
				const auto lodIdx = scene.lodIndices[hitId.instanceId];
				const auto& polygon = scene.lods[lodIdx].polygon;
				const u32 numVertices = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
				record_flux(polygon, importances[lodIdx], hitId.primId, numVertices, photonIt->position,
							get_luminance(photonIt->flux));
			}
			++photonIt;
		}
	} while(viewPathLen < params.maxPathLength);
}


CUDA_FUNCTION void sample_vis_importance(renderer::RenderBuffer<CURRENT_DEV>& outputBuffer,
										 const scene::SceneDescriptor<CURRENT_DEV>& scene,
										 const Pixel& coord, math::Rng& rng,
										 Importances<CURRENT_DEV>** importances,
										 const float maxImportance) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	VertexSample sample;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, nullptr, scene.camera.get(), coord, rng.next());

	scene::Point lastPosition = vertex.get_position();
	math::RndSet2_1 rnd{ rng.next(), rng.next() };
	float rndRoulette = math::sample_uniform(u32(rng.next()));
	if(walk(scene, vertex, rnd, rndRoulette, false, throughput, vertex, sample) == WalkResult::HIT) {
		const auto& hitpoint = vertex.get_position();
		const auto& hitId = vertex.get_primitive_id();
		const auto lodIdx = scene.lodIndices[hitId.instanceId];
		const auto& polygon = scene.lods[lodIdx].polygon;
		const u32 vertexCount = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
		const u32 vertexOffset = vertexCount == 3u ? 0u : (polygon.numTriangles * 3u);
		const u32 primIdx = vertexCount == 3u ? hitId.primId : (hitId.primId - polygon.numTriangles);

		float importance = 0.f;
		float distSqrSum = 0.f;
		for(u32 i = 0u; i < vertexCount; ++i)
			distSqrSum += ei::lensq(hitpoint - polygon.vertices[polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i]]);
		for(u32 i = 0u; i < vertexCount; ++i) {
			const auto vertexIndex = polygon.vertexIndices[vertexOffset + vertexCount * primIdx + i];
			const float distSqr = ei::lensq(hitpoint - polygon.vertices[vertexIndex]);
			importance += importances[lodIdx][vertexIndex].fluxImportance;
		}

		outputBuffer.contribute(coord, RenderTargets::RADIANCE, ei::Vec4{ importance / maxImportance });
	}
}

}}}}} // namespace mufflon::renderer::decimaters::silhouette::bpm