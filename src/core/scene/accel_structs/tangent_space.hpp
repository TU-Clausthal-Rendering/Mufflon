#pragma once

#include "intersection.hpp"

namespace mufflon { namespace scene { namespace accel_struct {



CUDA_FUNCTION TangentSpace tangent_space_geom_to_shader(const SceneDescriptor<CURRENT_DEV>& scene, const RayIntersectionResult& intersection) {
	// TODO: check for displacement mapping?
	const ObjectDescriptor<CURRENT_DEV>& object = scene.objects[scene.objectIndices[intersection.hitId.instanceId]];

	// Check for sphere as early as possible so that tri and quad threads have better divergence
	if(static_cast<u32>(intersection.hitId.primId) >= object.polygon.numTriangles + object.polygon.numQuads) {
		return TangentSpace{
			intersection.normal,
			intersection.normal,
			intersection.tangentX,
			intersection.tangentY
		};
	}

	if(object.polygon.numTriangles == 960) {
		__debugbreak();
	}

	// Compute the shading normal as the interpolated version of the per-vertex normals
	// Since the normal will be in world space, we need to transform it
	ei::Vec3 shadingNormal;
	const u32* vertexIndices = object.polygon.vertexIndices;
	const ei::Vec3* normals = object.polygon.normals;
	if(static_cast<u32>(intersection.hitId.primId) < object.polygon.numTriangles) {
		vertexIndices += 3u * intersection.hitId.primId;
		const ei::Vec2& barycentric = intersection.surfaceParams.barycentric;
		shadingNormal = barycentric.x * normals[vertexIndices[0u]]
			+ barycentric.y * normals[vertexIndices[1u]]
			+ (1.f - (barycentric.x + barycentric.y)) * normals[vertexIndices[2u]];
	} else {
		const u32 quadId = (intersection.hitId.primId - object.polygon.numTriangles);
		vertexIndices += 4u * quadId + 3u * object.polygon.numTriangles;
		const ei::Vec2& uv = intersection.surfaceParams.bilinear;
		shadingNormal = ei::bilerp(normals[vertexIndices[0u]],
								   normals[vertexIndices[1u]],
								   normals[vertexIndices[3u]],
								   normals[vertexIndices[2u]],
								   uv.x, uv.y);
	}

	// Transform the normal to world space (per-vertex normals are in object space)
	shadingNormal = ei::normalize(ei::Mat3x3{ scene.transformations[intersection.hitId.instanceId] } * shadingNormal);

	// Compute the shading tangents to make the systen orthonormal
	ei::Vec3 shadingTangentY = ei::normalize(ei::cross(intersection.tangentX, shadingNormal));
	// Flip the tangent (system is not guaranteed to be either left- or right-handed)
	if(ei::dot(shadingTangentY, intersection.tangentY) < 0)
		shadingTangentY *= -1.f;
	const ei::Vec3 shadingTangentX = ei::cross(shadingNormal, shadingTangentY);

	return TangentSpace{
		shadingNormal,
		intersection.normal,
		shadingTangentX,
		shadingTangentY
	};
}



}}} // namespace mufflon::scene::accel_struct