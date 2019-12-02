#pragma once

#include "util/types.hpp"
#include "util/punning.hpp"
#include "util/assert.hpp"
#include "core/export/api.h"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"

namespace mufflon {
namespace scene {
namespace accel_struct {

struct AccelStructInfo {
	struct Size
	{
		i32 offsetSpheres;
		i32 offsetQuads;
		i32 numVertices;
		i32 numPrimives;
		i32 bvhSize; // Number of ei::Vec4 in bvh.
	} sizes;
	struct InputArrays
	{
		ei::Vec3* meshVertices;
		ei::Vec2* meshUVs;
		i32* triIndices;
		i32* quadIndices;
		ei::Vec4* spheres;
	} inputs;
	struct OutputArrays {
		ei::Vec4* bvh;
		i32* primIds;
	} outputs;
};

inline CUDA_FUNCTION float int_bits_as_float(i32 v) {
#ifdef __CUDA_ARCH__
	return __int_as_float(v);
#else
	return util::pun<float>(v);
#endif // __CUDA_ARCH__
}

inline CUDA_FUNCTION i32 float_bits_as_int(float v) {
#ifdef __CUDA_ARCH__
	return __float_as_int(v);
#else
	return util::pun<i32>(v);
#endif // __CUDA_ARCH__
}

// Generic centroid overloads.
// This helps in generalizing the code of a builder
template < Device dev >
inline CUDA_FUNCTION ei::Vec3 get_centroid(const LodDescriptor<dev>& obj, i32 primIdx) {
	// Primitve order: Trianges, Quads, Spheres -> idx determines the case
	i32 spheresOffset = obj.polygon.numQuads + obj.polygon.numTriangles;
	if(primIdx >= spheresOffset)
		return obj.spheres.spheres[primIdx - spheresOffset].center;
	if(primIdx >= i32(obj.polygon.numTriangles)) {
		i32 quadId = (primIdx - obj.polygon.numTriangles) * 4 + obj.polygon.numTriangles * 3;
		return (obj.polygon.vertices[obj.polygon.vertexIndices[quadId  ]]
			  + obj.polygon.vertices[obj.polygon.vertexIndices[quadId+1]]
			  + obj.polygon.vertices[obj.polygon.vertexIndices[quadId+2]]
			  + obj.polygon.vertices[obj.polygon.vertexIndices[quadId+3]]) / 4.0f;
	}
	i32 triId = primIdx * 3;
	return (obj.polygon.vertices[obj.polygon.vertexIndices[triId  ]]
		  + obj.polygon.vertices[obj.polygon.vertexIndices[triId+1]]
		  + obj.polygon.vertices[obj.polygon.vertexIndices[triId+2]]) / 3.0f;
}

template < Device dev >
inline CUDA_FUNCTION ei::Vec3 get_centroid(const SceneDescriptor<dev>& scene, i32 primIdx) {
	const i32 objIdx = scene.lodIndices[primIdx];
	const ei::Mat3x4 instanceToWorld = scene.compute_instance_to_world_transformation(primIdx);
	// Transform the center only (no need to compute the full bounding box).
	return transform(center(scene.aabbs[objIdx]), instanceToWorld);
}

// Generic bounding box overloads.
// This helps in generalizing the code of a builder
template < Device dev >
inline CUDA_FUNCTION ei::Box get_bounding_box(const LodDescriptor<dev>& obj, i32 idx) {
	// Primitve order: Trianges, Quads, Spheres -> idx determines the case
	i32 spheresOffset = obj.polygon.numQuads + obj.polygon.numTriangles;
	if(idx >= spheresOffset)
		return ei::Box(obj.spheres.spheres[idx - spheresOffset]);
	if(idx >= i32(obj.polygon.numTriangles)) {
		i32 quadId = (idx - obj.polygon.numTriangles) * 4 + obj.polygon.numTriangles * 3;
		return ei::Box(obj.polygon.vertices[obj.polygon.vertexIndices[quadId  ]],
					   obj.polygon.vertices[obj.polygon.vertexIndices[quadId+1]],
					   obj.polygon.vertices[obj.polygon.vertexIndices[quadId+2]],
					   obj.polygon.vertices[obj.polygon.vertexIndices[quadId+3]]);
	}
	i32 triId = idx * 3;
	return ei::Box(obj.polygon.vertices[obj.polygon.vertexIndices[triId  ]],
				   obj.polygon.vertices[obj.polygon.vertexIndices[triId+1]],
				   obj.polygon.vertices[obj.polygon.vertexIndices[triId+2]]);
}

template < Device dev >
inline CUDA_FUNCTION ei::Box get_bounding_box(const SceneDescriptor<dev>& scene, i32 idx) {
	i32 objIdx = scene.lodIndices[idx];
	const ei::Mat3x4 instanceToWorld = scene.compute_instance_to_world_transformation(idx);
	return transform(scene.aabbs[objIdx], instanceToWorld);
}

}}} // namespace mufflon::scene::accel_struct
