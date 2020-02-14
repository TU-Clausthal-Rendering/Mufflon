#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/export/core_api.h"
#include "core/scene/descriptors.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>

namespace mufflon { namespace scene {

inline CUDA_FUNCTION ei::Triangle get_triangle(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon,
											   const u32 primId) {

	const ei::IVec3 indices{
			polygon.vertexIndices[3u * primId + 0],
			polygon.vertexIndices[3u * primId + 1],
			polygon.vertexIndices[3u * primId + 2]
	};
	return ei::Triangle{
		polygon.vertices[indices.x],
		polygon.vertices[indices.y],
		polygon.vertices[indices.z]
	};
}
inline CUDA_FUNCTION ei::Tetrahedron get_quad(const scene::PolygonsDescriptor<CURRENT_DEV>& polygon,
											  const u32 primId) {
	const u32 vertexOffset = polygon.numTriangles * 3u;
	const u32 primIdx = primId - polygon.numTriangles;
	const ei::IVec4 indices{
			polygon.vertexIndices[vertexOffset + 4u * primIdx + 0],
			polygon.vertexIndices[vertexOffset + 4u * primIdx + 1],
			polygon.vertexIndices[vertexOffset + 4u * primIdx + 2],
			polygon.vertexIndices[vertexOffset + 4u * primIdx + 3]
	};
	return ei::Tetrahedron{
		polygon.vertices[indices.x],
		polygon.vertices[indices.y],
		polygon.vertices[indices.z],
		polygon.vertices[indices.w]
	};
}

inline CUDA_FUNCTION float compute_area(const ei::Triangle& triangle, const ei::Mat3x4& transform) noexcept {
	const auto a = ei::transform(triangle.v0, transform);
	const auto b = ei::transform(triangle.v1, transform);
	const auto c = ei::transform(triangle.v2, transform);

	const auto normal = ei::cross(b - a, c - a);
	return 0.5f * ei::len(normal);
}
inline CUDA_FUNCTION float compute_area(const ei::Tetrahedron& quad, const ei::Mat3x4& transform) noexcept {
	// Approximate quad as two triangles
	const ei::Triangle a{ quad.v0, quad.v1, quad.v2 };
	const ei::Triangle b{ quad.v0, quad.v2, quad.v3 };
	return compute_area(a, transform) + compute_area(b, transform);
}

inline CUDA_FUNCTION float compute_area_instance_transformed(const scene::SceneDescriptor<CURRENT_DEV>& scene,
															 const scene::PolygonsDescriptor<CURRENT_DEV>& polygon,
															 const scene::PrimitiveHandle hitId) noexcept {
	const u32 vertexCount = hitId.primId < (i32)polygon.numTriangles ? 3u : 4u;
	const auto instToWorld = scene.compute_instance_to_world_transformation(hitId.instanceId);

	if(vertexCount == 3u)
		return mufflon::scene::compute_area(mufflon::scene::get_triangle(polygon, static_cast<u32>(hitId.primId)), instToWorld);
	else
		return mufflon::scene::compute_area(mufflon::scene::get_quad(polygon, static_cast<u32>(hitId.primId)), instToWorld);

}

}} // namespace mufflon::scene