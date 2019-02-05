#pragma once

#include "intersection.hpp"

namespace mufflon { namespace scene { namespace accel_struct {



CUDA_FUNCTION TangentSpace tangent_space_geom_to_shader(const SceneDescriptor<CURRENT_DEV>& scene, const RayIntersectionResult& intersection) {
	// TODO: check for displacement mapping?
	const LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[intersection.hitId.instanceId]];

	// Check for sphere as early as possible so that tri and quad threads have better divergence
	if(static_cast<u32>(intersection.hitId.primId) >= object.polygon.numTriangles + object.polygon.numQuads) {
		return TangentSpace{
			intersection.normal,
			intersection.normal,
			intersection.tangentX,
			intersection.tangentY
		};
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
	const ei::Vec3& scale = scene.scales[intersection.hitId.instanceId];
	const ei::Mat3x3 rotation = ei::Mat3x3{ scene.transformations[intersection.hitId.instanceId] };
	shadingNormal = ei::normalize(rotation * (shadingNormal / scale));

	// Compute orthonormal shading tangents
	// Gram-Schmidt
	const ei::Vec3 shadingTangentX = normalize(
		intersection.tangentX - shadingNormal * dot(intersection.tangentX, shadingNormal));
	const ei::Vec3 shadingTangentY = cross(shadingNormal, shadingTangentX);
	// Flip the tangent (system is not guaranteed to be either left- or right-handed)
	// DISABLED: Likely not required for any shading model...
	//if(dot(shadingTangentY, intersection.tangentY) < 0)
	//	shadingTangentY = -shadingTangentY;

	return TangentSpace{
		shadingNormal,
		intersection.normal,
		shadingTangentX,
		shadingTangentY
	};
}



}}} // namespace mufflon::scene::accel_struct