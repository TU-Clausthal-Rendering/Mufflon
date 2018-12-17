#pragma once

#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/materials/material.hpp"

namespace mufflon { namespace scene { namespace materials {

CUDA_FUNCTION scene::materials::MediumHandle get_point_medium(const scene::SceneDescriptor<CURRENT_DEV>& scene, const ei::Vec3& pos) {
	mAssert(scene.objects[0u].polygon.numVertices > 0u || scene.objects[0u].spheres.numSpheres > 0u);
	// Shoot a ray to a point in the scene (any vertex suffices)
	ei::Vec3 vertex;
	if(scene.objects[0u].polygon.numVertices > 0u)
		vertex = scene.objects[0u].polygon.vertices[0u];
	else
		vertex = scene.objects[0u].spheres.spheres[0u].center;

	ei::Vec3 dir = vertex - pos;
	const float length = ei::len(dir);
	dir *= 1.f / dir;
	ei::Ray ray{ pos, dir };
	scene::accel_struct::RayIntersectionResult res;
	scene::accel_struct::first_intersection_scene_lbvh<CURRENT_DEV>(scene, ray, 0xFFFFFFFF00000000ull, length + 1.f, res);
	mAssert(res.hitPrimId != (u64)-1ll);
	// From the intersection we get the primitive, from which we can look up the material
	const u32 INSTANCE_ID = res.hitPrimId >> 32u;
	const u32 PRIMITIVE_ID = static_cast<u32>(res.hitPrimId & 0xFFFFFFFF);

	const scene::ObjectDescriptor<CURRENT_DEV>& object = scene.objects[scene.objectIndices[INSTANCE_ID]];
	const u32 FACE_COUNT = object.polygon.numTriangles + object.polygon.numQuads;
	scene::MaterialIndex matIdx;
	if(PRIMITIVE_ID < FACE_COUNT)
		matIdx = object.polygon.matIndices[PRIMITIVE_ID];
	else
		matIdx = object.spheres.matIndices[PRIMITIVE_ID - FACE_COUNT];

	return scene.get_material(matIdx).get_medium(ei::dot(dir, res.normal));
}

}}} // namespace mufflon::scene::materials