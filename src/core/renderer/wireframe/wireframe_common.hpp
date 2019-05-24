#pragma once

#include "wireframe_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <ei/vector.hpp>

namespace mufflon { namespace renderer {

using PtPathVertex = PathVertex<VertexExtension>;

namespace {

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<CURRENT_DEV>& scene, const i32 instanceId,
											   const ei::IVec3& indices, const ei::Vec3& hitpoint) {
	// Compute the projected points on the triangle lines
	const auto& vertices = scene.lods[scene.lodIndices[instanceId]].polygon.vertices;
	const auto A = transform(vertices[indices.x], scene.instanceToWorld[instanceId]);
	const auto B = transform(vertices[indices.y], scene.instanceToWorld[instanceId]);
	const auto C = transform(vertices[indices.z], scene.instanceToWorld[instanceId]);
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

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<CURRENT_DEV>& scene, const i32 instanceId,
											   const ei::IVec4& indices, const ei::Vec3& hitpoint) {
	// Compute the projected points on the quad lines
	const auto& vertices = scene.lods[scene.lodIndices[instanceId]].polygon.vertices;
	const auto A = transform(vertices[indices.x], scene.instanceToWorld[instanceId]);
	const auto B = transform(vertices[indices.y], scene.instanceToWorld[instanceId]);
	const auto C = transform(vertices[indices.w], scene.instanceToWorld[instanceId]);
	const auto D = transform(vertices[indices.z], scene.instanceToWorld[instanceId]);
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

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<CURRENT_DEV>& scene, const i32 instanceId,
											   const u32 index, const ei::Vec3& hitpoint, const ei::Vec3& incident) {
	const auto& sphere = scene.lods[scene.lodIndices[instanceId]].spheres.spheres[index];
	const auto center = transform(sphere.center, scene.instanceToWorld[instanceId]);
	// First we compute the vector pointing to the edge of the sphere from our point-of-view
	const auto centerToHit = hitpoint - center;
	const auto down = ei::cross(centerToHit, incident);
	const auto centerToRim = ei::normalize(ei::cross(incident, down));
	// Now scale with radius -> done (TODO: scale radius? probably, but don't wanna check right now)
	return centerToRim * sphere.radius;
}

} // namespace 

CUDA_FUNCTION void sample_wireframe(RenderBuffer<CURRENT_DEV>& outputBuffer,
									scene::SceneDescriptor<CURRENT_DEV>& scene,
									const WireframeParameters& params, math::Rng& rng, const Pixel& coord) {
	constexpr ei::Vec3 borderColor{ 1.f };

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, scene.camera.get(), coord, rng.next());
	VertexSample sample = vertex.sample(scene.media, math::RndSet2_1{ rng.next(), rng.next() }, false);
	ei::Ray ray{ sample.origin, sample.excident };
	float totalDistance = 0.f;

	// Arbitrary upper limit to avoid complete hangings
	for(int i = 0; i < params.maxTraceDepth; ++i) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection<CURRENT_DEV, false>(scene, ray, vertex.get_geometric_normal(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(scene.lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const auto& hitId = nextHit.hitId;
			const auto& lod = scene.lods[scene.lodIndices[hitId.instanceId]];
			const auto& poly = lod.polygon;
			const float hitDist = nextHit.distance < 0.001f ? 0.001f : nextHit.distance;
			const auto& hit = ray.origin + ray.direction * hitDist;
			totalDistance += hitDist;

			ei::Vec3 closestLinePoint;
			if(static_cast<u32>(nextHit.hitId.primId) < poly.numTriangles) {
				const ei::IVec3 indices{
					poly.vertexIndices[3 * hitId.primId + 0],
					poly.vertexIndices[3 * hitId.primId + 1],
					poly.vertexIndices[3 * hitId.primId + 2]
				};
				closestLinePoint = computeClosestLinePoint(scene, hitId.instanceId, indices, hit);
			} else if(static_cast<u32>(nextHit.hitId.primId) < (poly.numTriangles + poly.numQuads)) {
				const u32 quadId = static_cast<u32>(nextHit.hitId.primId) - poly.numTriangles;
				const ei::IVec4 indices{
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 0],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 1],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 2],
					poly.vertexIndices[3 * poly.numTriangles + 4 * quadId + 3]
				};
				closestLinePoint = computeClosestLinePoint(scene, hitId.instanceId, indices, hit);
			} else {
				closestLinePoint = computeClosestLinePoint(scene, hitId.instanceId, hitId.primId, hit, ray.direction);
			}

			// If the point is within x pixels of the line we paint
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
			// Compute the projected length of the line between face line and hitpoint
			const auto lineSegment = closestLinePoint - vertex.get_position();
			const auto projectedLineSegment = lineSegment - ei::dot(lineSegment, ray.direction) * ray.direction;
			const float sqDistThreshold = ei::sq(params.lineWidth * pixelSize);
			// Only draw it if it's below the threshold (expressed in pixels)
			if(ei::lensq(projectedLineSegment) > sqDistThreshold) {
				ray.origin = ray.origin + ray.direction * hitDist;
			} else {
				outputBuffer.contribute(coord, throughput, borderColor,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
				break;
			}
		}
	}
}

}} // namespace mufflon::renderer