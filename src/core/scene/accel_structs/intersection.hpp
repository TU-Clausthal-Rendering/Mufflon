#pragma once

#include "accel_structs_commons.hpp"
#include "core/scene/handles.hpp"
#include "core/memory/residency.hpp"
#include "core/scene/descriptors.hpp"

#include <ei/3dtypes.hpp>

// TODO: use u32 for primiIds.

namespace mufflon { namespace scene { namespace accel_struct {

// The surface coordinates of the hitpoint, depending on primitive type
union SurfaceParametrization {
	ei::Vec2 barycentric;	// Triangle
	ei::Vec2 bilinear;		// Quad
	ei::Vec2 st;			// Sphere

	__host__ __device__ SurfaceParametrization() {}
};

struct RayIntersectionResult {
	float distance;
	PrimitiveHandle hitId;
	ei::Vec3 normal;		// Normalized geometric normal
	ei::Vec3 tangentX;		// Normalized geometric tangent in texture U direction
	ei::Vec2 uv;
	SurfaceParametrization surfaceParams;
};

struct RayInfo {
	ei::Ray ray;
	i32 startPrimId;
	float tmin;
	float tmax;
};

template < Device dev = CURRENT_DEV >
__host__ __device__
bool any_intersection(
	const SceneDescriptor<dev>& scene,
	scene::Point a,
	scene::Point b,
	const scene::Direction& geoNormalA,
	const scene::Direction& geoNormalB,
	const scene::Direction& connectionDirAtoB
);

template < Device dev = CURRENT_DEV, bool alphatest = true >
__host__ __device__
RayIntersectionResult first_intersection(
	const SceneDescriptor<dev>& scene,
	ei::Ray& ray,
	const ei::Vec3& geoNormal,
	const float tmax
);

}}} // namespace mufflon::scene::accel_struct
