#pragma once

#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/accel_structs/accel_structs_commons.hpp"
#include "core/scene/materials/material.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION scene::materials::MediumHandle get_point_medium(const scene::SceneDescriptor<CURRENT_DEV>& scene, const ei::Vec3& pos) {
	mAssert(scene.objects[0u].polygon.numVertices > 0u || scene.objects[0u].spheres.numSpheres > 0u);
	// Shoot a ray to a point in the scene (any surface suffices)
	// We need to transform the vertex from object to world space
	const ei::Vec3 vertex = scene.transformations[0u] 
		* ei::Vec4{accel_struct::get_centroid(scene.objects[scene.objectIndices[0u]], 0), 1.0f};

	ei::Vec3 dir = vertex - pos;
	const float length = ei::len(dir);
	dir *= 1.f / length;
	ei::Ray ray{ pos, dir };
	auto res = accel_struct::first_intersection_scene_lbvh<CURRENT_DEV>(scene, ray, { -1l, -1l }, length + 1.f);
	mAssert(res.hitId.instanceId != -1l);
	// From the intersection we get the primitive, from which we can look up the material
	const i32 instanceId = res.hitId.instanceId;
	const u32 primitiveId = res.hitId.get_primitive_id();

	const scene::ObjectDescriptor<CURRENT_DEV>& object = scene.objects[scene.objectIndices[instanceId]];
	const u32 faceCount = object.polygon.numTriangles + object.polygon.numQuads;
	scene::MaterialIndex matIdx;
	if(primitiveId < faceCount)
		matIdx = object.polygon.matIndices[primitiveId];
	else
		matIdx = object.spheres.matIndices[primitiveId - faceCount];

	return scene.get_material(matIdx).get_medium(ei::dot(dir, res.normal));
}

}}} // namespace mufflon::scene::materials