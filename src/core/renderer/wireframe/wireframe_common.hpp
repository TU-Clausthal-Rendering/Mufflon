#pragma once

#include "wireframe_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/path_util.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace renderer {

using PtPathVertex = PathVertex<VertexExtension>;

namespace {

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const ei::Vec3* vertices, const ei::IVec3& indices,
											   const ei::Mat3x4& instanceToWorld, const ei::Vec3& hitpoint) {
	// Compute the projected points on the triangle lines
	const auto A = transform(vertices[indices.x], instanceToWorld);
	const auto B = transform(vertices[indices.y], instanceToWorld);
	const auto C = transform(vertices[indices.z], instanceToWorld);
	const auto AB = B - A;
	const auto AC = C - A;
	const auto BC = C - B;
	const auto AP = hitpoint - A;
	const auto BP = hitpoint - B;
	const auto onAB = A + ei::dot(AP, AB) / ei::lensq(AB) * AB;
	const auto onAC = A + ei::dot(AP, AC) / ei::lensq(AC) * AC;
	const auto onBC = B + ei::dot(BP, BC) / ei::lensq(BC) * BC;

	// Determine the point closest to the hitpoint
	const auto distAB = ei::lensq(onAB - hitpoint);
	const auto distAC = ei::lensq(onAC - hitpoint);
	const auto distBC = ei::lensq(onBC - hitpoint);
	ei::Vec3 closestLinePoint;
	if(distAB <= distAC && distAB <= distBC)
		return onAB;
	else if(distAC <= distAB && distAC <= distBC)
		return onAC;
	else
		return onBC;
}

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const ei::Vec3* vertices, const ei::IVec4& indices,
											   const ei::Mat3x4& instanceToWorld, const ei::Vec3& hitpoint) {
	// Compute the projected points on the quad lines
	const auto A = transform(vertices[indices.x], instanceToWorld);
	const auto B = transform(vertices[indices.y], instanceToWorld);
	const auto C = transform(vertices[indices.w], instanceToWorld);
	const auto D = transform(vertices[indices.z], instanceToWorld);
	const auto AB = B - A;
	const auto AC = C - A;
	const auto BD = D - B;
	const auto CD = D - C;
	const auto AP = hitpoint - A;
	const auto BP = hitpoint - B;
	const auto CP = hitpoint - C;
	const auto onAB = A + ei::dot(AP, AB) / ei::lensq(AB) * AB;
	const auto onAC = A + ei::dot(AP, AC) / ei::lensq(AC) * AC;
	const auto onBD = B + ei::dot(BP, BD) / ei::lensq(BD) * BD;
	const auto onCD = C + ei::dot(CP, CD) / ei::lensq(CD) * CD;

	// Determine the point closest to the hitpoint
	const auto distAB = ei::lensq(onAB - hitpoint);
	const auto distAC = ei::lensq(onAC - hitpoint);
	const auto distBD = ei::lensq(onBD - hitpoint);
	const auto distCD = ei::lensq(onCD - hitpoint);
	ei::Vec3 closestLinePoint;
	if(distAB <= distAC && distAB <= distBD && distAB <= distCD)
		return onAB;
	else if(distAC <= distAB && distAC <= distBD && distAC <= distCD)
		return onAC;
	else if(distBD <= distAB && distBD <= distAC && distBD <= distCD)
		return onBD;
	else
		return onCD;
}

CUDA_FUNCTION float computeDistToRim(const ei::Sphere* spheres, const u32 index,
									 const ei::Mat3x4& instanceToWorld, const ei::Vec3& hitpoint,
									 const ei::Vec3& incident, const ei::Vec3& origin) {
	const auto& sphere = spheres[index];
	const auto center = transform(sphere.center, instanceToWorld);
	// We use the angle between incident and center-hitpoint as the indicator of
	// proximity to tangential ray (90° == tangent, 0° == max. non-tangentness
	const auto hitToCenter = ei::normalize(center - hitpoint);
	const float angleIntersectCenter = ei::dot(hitToCenter, incident);
	return sphere.radius * angleIntersectCenter;
}

} // namespace 

CUDA_FUNCTION void sample_wireframe(WireframeTargets::RenderBufferType<CURRENT_DEV>& outputBuffer,
									scene::SceneDescriptor<CURRENT_DEV>& scene,
									const WireframeParameters& params, math::Rng& rng, const Pixel& coord) {
	constexpr ei::Vec3 borderColor{ 1.f };

	PtPathVertex vertex;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, scene.camera.get(), coord, rng.next());
	VertexSample sample = vertex.sample(scene.aabb, scene.media, math::RndSet2_1{ rng.next(), rng.next() }, false);
	ei::Ray ray{ sample.origin, sample.excident };
	float totalDistance = 0.f;

	// Arbitrary upper limit to avoid complete hangings
	for(int i = 0; i < params.maxTraceDepth; ++i) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection<CURRENT_DEV, false>(scene, ray, vertex.get_geometric_normal(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(scene.lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				outputBuffer.contribute<BorderTarget>(coord, background.value);
			}
			break;
		} else {
			const auto& hitId = nextHit.hitId;
			const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
			const auto& poly = lod.polygon;
			const float hitDist = nextHit.distance < 0.001f ? 0.001f : nextHit.distance;
			const auto& hit = ray.origin + ray.direction * hitDist;
			totalDistance += hitDist;

			const auto& camParams = scene.camera.get();
			float pixelSize = 0.f;
			switch(camParams.type) {
				case cameras::CameraModel::PINHOLE:
				{
					const auto& pinholeParams = static_cast<const cameras::PinholeParams&>(camParams);
					pixelSize = totalDistance * pinholeParams.tanVFov / static_cast<float>(pinholeParams.resolution.y);
				}	break;
				case cameras::CameraModel::FOCUS:	// TODO: does this make sense?
				case cameras::CameraModel::ORTHOGRAPHIC:
				default:
					mAssertMsg(false, "Unknown or invalid camera model");
			}

			float projDistToRim;
			const ei::Mat3x4 instanceToWorld = scene.compute_instance_to_world_transformation(nextHit.hitId.instanceId);
			if(static_cast<u32>(nextHit.hitId.primId) < poly.numTriangles) {
				const ei::IVec3 indices{
					poly.vertexIndices[3 * hitId.primId + 0],
					poly.vertexIndices[3 * hitId.primId + 1],
					poly.vertexIndices[3 * hitId.primId + 2]
				};
				const ei::Vec3 closestLinePoint = computeClosestLinePoint(lod.polygon.vertices,
																		  indices, instanceToWorld, hit);
				const auto lineSegment = closestLinePoint - vertex.get_position();
				const auto projectedLineSegment = lineSegment - ei::dot(lineSegment, ray.direction) * ray.direction;
				projDistToRim = ei::len(projectedLineSegment);
			} else if(static_cast<u32>(nextHit.hitId.primId) < (poly.numTriangles + poly.numQuads)) {
				const u32 quadId = static_cast<u32>(nextHit.hitId.primId) - poly.numTriangles;
				const ei::IVec4 indices{
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 0],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 1],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 2],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 3]
				};
				const ei::Vec3 closestLinePoint = computeClosestLinePoint(lod.polygon.vertices,
																		  indices, instanceToWorld, hit);
				const auto lineSegment = closestLinePoint - vertex.get_position();
				const auto projectedLineSegment = lineSegment - ei::dot(lineSegment, ray.direction) * ray.direction;
				projDistToRim = ei::len(projectedLineSegment);
			} else {
				// TODO: this code goes to 0 when going towards being tangential, however
				// it doesn't do it fast enough/in a correct way; the applied factor
				// here is a workaround to get roughly the same line width as triangles
				// and quads
				projDistToRim = 0.05f * computeDistToRim(lod.spheres.spheres, hitId.primId,
														 instanceToWorld, hit, ray.direction,
														 sample.origin);
			}
			const float distThreshold = params.lineWidth * pixelSize;
			// Only draw it if it's below the threshold (expressed in pixels)
			if(projDistToRim > distThreshold) {
				ray.origin = ray.origin + ray.direction * hitDist;
			} else {
				outputBuffer.contribute<BorderTarget>(coord, borderColor);
				break;
			}
		}
	}
}

}} // namespace mufflon::renderer