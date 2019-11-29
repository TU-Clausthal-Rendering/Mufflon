#pragma once

#include "intersection.hpp"
#include "core/math/curvature.hpp"

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
			cross(intersection.normal, intersection.tangentX)
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
	const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[intersection.hitId.instanceId] });
	shadingNormal = ei::normalize(rotationInvScale * shadingNormal);

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


/*
 * Compute the mean curvature of a primitive.
 * The result is mostly correct, but very expensive due to lots of transformations
 * which are necessary to support non-uniform scaling.
 * The curvature is constant over the face.
 */
CUDA_FUNCTION float compute_face_curvature(
		const SceneDescriptor<CURRENT_DEV>& scene,
		const PrimitiveHandle& hitId,
		const Direction& geoNormal) {
	const LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[hitId.instanceId]];
	const u32* vertexIndices = object.polygon.vertexIndices;
	const Direction* normals = object.polygon.normals;
	const Point* positions = object.polygon.vertices;
	if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles) {
		// Triangle
		vertexIndices += 3u * hitId.primId;
		const ei::Mat3x4 instanceToWorld{ ei::invert(ei::Mat4x4{ scene.instanceToWorld[hitId.instanceId] }) };
		const ei::Mat3x3 transform{ instanceToWorld };
		const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
		auto c = math::compute_curvature(geoNormal, transform * positions[vertexIndices[0]],
										 transform * positions[vertexIndices[1]],
										 transform * positions[vertexIndices[2]],
										 normalize(rotationInvScale * normals[vertexIndices[0]]),
										 normalize(rotationInvScale * normals[vertexIndices[1]]),
										 normalize(rotationInvScale * normals[vertexIndices[2]]));
		return c.get_mean_curvature();
	} if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
		// Quad
		const u32 quadId = (hitId.primId - object.polygon.numTriangles);
		vertexIndices += 4u * quadId + 3u * object.polygon.numTriangles;
		const ei::Mat3x4 instanceToWorld{ ei::invert(ei::Mat4x4{ scene.instanceToWorld[hitId.instanceId] }) };
		const ei::Mat3x3 transform { instanceToWorld };
		const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
		auto c = math::compute_curvature(geoNormal, transform * positions[vertexIndices[0]],
										 transform * positions[vertexIndices[1]],
										 transform * positions[vertexIndices[2]],
										 transform * positions[vertexIndices[3]],
										 normalize(rotationInvScale * normals[vertexIndices[0]]),
										 normalize(rotationInvScale * normals[vertexIndices[1]]),
										 normalize(rotationInvScale * normals[vertexIndices[2]]),
										 normalize(rotationInvScale * normals[vertexIndices[3]]));
		return c.get_mean_curvature();
	} else {
		// Sphere
		// Assume uniform scaling from instancing:
		float scale = len(ei::Vec3{scene.worldToInstance[hitId.instanceId]});
		return scale / object.spheres.spheres[hitId.primId].radius;
	}
}


/*
 * Interpolate the mean curvature from precomputed vertex-curvatures.
 * Requires: scene.compute_curvature() before the descriptor is fetched.
 *
 * Dependent on the tessellation this may have some error:
 *	- low vertex adjacency -> high numerical error
 *	- interpolation from high curvature vertices over large regions
 *	  e.g. beveled box.
 */
CUDA_FUNCTION float fetch_curvature(const SceneDescriptor<CURRENT_DEV>& scene,
		const PrimitiveHandle& hitId, const ei::Vec2& surfParam,
		const Direction& geoNormal) {
	const LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[hitId.instanceId]];
	const u32* vertexIndices = object.polygon.vertexIndices;
	const float* curvature = static_cast<const float*>(object.polygon.vertexAttributes[0]);
	// The precomputed curvature values do not include the instance transformation,
	// which might have a non-uniform scaling.
	// The problem is that there are not enough information stored to reconstruct
	// the scaling. It would be nesessary to know the entire Weingarten matrix
	// and the tangent space which was used to compute it.
	// Best approximative guess: get average scaling through the cubic root
	// of the determinant.
	float scale = pow(ei::abs(determinant(ei::Mat3x3{scene.worldToInstance[hitId.instanceId]})), 1/3.0f);
	if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles) {
		// Triangle
		vertexIndices += 3u * hitId.primId;
		return scale *
			 (curvature[vertexIndices[0]] * surfParam.x
			+ curvature[vertexIndices[1]] * surfParam.y
			+ curvature[vertexIndices[2]] * (1.0f - surfParam.x - surfParam.y));
	} else if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
		// Quad
		const u32 quadId = (hitId.primId - object.polygon.numTriangles);
		vertexIndices += 4u * quadId + 3u * object.polygon.numTriangles;
		return scale * ei::bilerp(
			curvature[vertexIndices[0]], curvature[vertexIndices[1]],
			curvature[vertexIndices[3]], curvature[vertexIndices[2]], surfParam.x, surfParam.y);
	} else {
		// Sphere
		// Assume uniform scaling from instancing:
	//	float scale = len(ei::Vec3{scene.worldToInstance[hitId.instanceId]});
		return scale / object.spheres.spheres[hitId.primId].radius;
	}
}


}}} // namespace mufflon::scene::accel_struct