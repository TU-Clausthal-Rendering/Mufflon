#pragma once

#include "core/scene/descriptors.hpp"
#include "core/scene/types.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/accel_structs/accel_structs_commons.hpp"
#include "core/scene/materials/material_definitions.hpp"

namespace mufflon { namespace scene { namespace materials {

inline CUDA_FUNCTION scene::materials::MediumHandle get_point_medium(const scene::SceneDescriptor<CURRENT_DEV>& scene, const ei::Vec3& pos) {
	mAssert(scene.lods[scene.lodIndices[scene.validInstanceIndex]].polygon.numVertices > 0u
		|| scene.lods[scene.lodIndices[scene.validInstanceIndex]].spheres.numSpheres > 0u);
	// Shoot a ray to a point in the scene (any surface suffices)
	// We need to transform the vertex from object to world space
	const Point objSpaceCenter = accel_struct::get_centroid(scene.lods[scene.lodIndices[scene.validInstanceIndex]], 0);
	const ei::Mat3x4 instanceToWorld = scene.compute_instance_to_world_transformation(scene.validInstanceIndex);
	const Point vertex = transform(objSpaceCenter, instanceToWorld);

	Direction dir = vertex - pos;
	const float length = ei::len(dir);
	dir /= length;
	ei::Ray ray{ pos, dir };
	// Ignore alpha testing
	auto res = accel_struct::first_intersection<CURRENT_DEV, false>(scene, ray, ei::Vec3{0.0f}, length + 1.f);
	mAssert(res.hitId.instanceId != -1l);
	// From the intersection we get the primitive, from which we can look up the material
	const i32 instanceId = res.hitId.instanceId;
	const u32 primitiveId = static_cast<u32>(res.hitId.primId);

	const scene::LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[instanceId]];
	const u32 faceCount = object.polygon.numTriangles + object.polygon.numQuads;
	scene::MaterialIndex matIdx;
	if(primitiveId < faceCount)
		matIdx = object.polygon.matIndices[primitiveId];
	else
		matIdx = object.spheres.matIndices[primitiveId - faceCount];

	return scene.get_material(matIdx).get_medium(-ei::dot(dir, res.normal));
}

}}} // namespace mufflon::scene::materials
