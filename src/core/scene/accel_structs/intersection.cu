#include "intersection.hpp"
#include "accel_structs_commons.hpp"
#include "lbvh.hpp"
#include "util/types.hpp"
#include "core/scene/textures/interface.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <ei/3dintersection.hpp>

namespace mufflon { namespace scene { namespace accel_struct {

namespace {

constexpr float SCENE_SCALE_EPS = 1e-4f;

CUDA_FUNCTION __forceinline void add_epsilon(ei::Vec3& rayOrigin, const ei::Vec3& dir, const ei::Vec3& geoNormal) {
	ei::Vec3 offset = geoNormal * SCENE_SCALE_EPS;
	if(dot(geoNormal, dir) >= 0.0f)
		rayOrigin += offset;
	else
		rayOrigin -= offset;
}


#define STACK_SIZE              64*2+1         // Size of the traversal stack in local memory.
#define OBJ_STACK_SIZE          64+1           // Size of the traversal stack in local memory.
enum : i32 {
	EntrypointSentinel = (i32)0xFFFFFFFF,   // Bottom-most stack entry, indicating the end of traversal.
	IGNORE_ID = (i32)0xFFFFFFFF
};

// Experimentally determined best mix of float/i32/video minmax instructions for Kepler.
__device__ __forceinline__ i32   min_min(i32 a, i32 b, i32 c) { i32 v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __forceinline__ i32   min_max(i32 a, i32 b, i32 c) { i32 v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __forceinline__ i32   max_min(i32 a, i32 b, i32 c) { i32 v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __forceinline__ i32   max_max(i32 a, i32 b, i32 c) { i32 v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __forceinline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __forceinline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __forceinline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __forceinline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __forceinline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __forceinline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }


CUDA_FUNCTION bool intersect(const ei::Box& bb,
	const ei::FastRay& ray,
	const float tmax, float& cmin) {//, float& cmax) {
#ifdef __CUDA_ARCH__
	ei::Vec3 lo = bb.min * ray.invDirection - ray.oDivDir;
	ei::Vec3 hi = bb.max * ray.invDirection - ray.oDivDir;
	cmin = spanBeginKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, 0.0f);
	const float cmax = spanEndKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);
	return cmin <= cmax;
#else
	float t0 = bb.min.x * ray.invDirection.x - ray.oDivDir.x;
	float t1 = bb.max.x * ray.invDirection.x - ray.oDivDir.x;
	cmin = ei::min(t0, t1);
	float cmax = ei::max(t0, t1);
	if (cmax < 0.0f || cmin > tmax) return false;
	t0 = bb.min.y * ray.invDirection.y - ray.oDivDir.y;
	t1 = bb.max.y * ray.invDirection.y - ray.oDivDir.y;
	float min2 = ei::min(t0, t1);
	float max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	if (cmax < 0.0f || cmin > tmax || cmin > cmax) return false;
	t0 = bb.min.z * ray.invDirection.z - ray.oDivDir.z;
	t1 = bb.max.z * ray.invDirection.z - ray.oDivDir.z;
	min2 = ei::min(t0, t1);
	max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	return (cmax >= 0.0f) && (cmin <= tmax) && (cmin <= cmax);
#endif // __CUDA_ARCH__
}

// Partial abs: Get the absolute value except for 0 (in case of -0, the result will be -0);
CUDA_FUNCTION __forceinline__ float pabs(const float x) {
	return x < 0.0f ? -x : x;
}

// Helper functions for the intersection test
CUDA_FUNCTION __forceinline__ float computeU(const float v, const float A1, const float A2,
											 const float B1, const float B2, const float C1,
											 const float C2, const float D1, const float D2) {
	const float a = v * A2 + B2;
	const float b = v * (A2 - A1) + B2 - B1;
	if(ei::abs(b) >= ei::abs(a))
		return (v*(C1 - C2) + D1 - D2) / b;
	else
		return (-v * C2 - D2) / a;
}

// Quad intersection test
// Implementation from http://www.sci.utah.edu/~kpotter/publications/ramsey-2004-RBPI.pdf
//CUDA_FUNCTION float intersectQuad(const ei::Tetrahedron& quad, const ei::FastRay& ray, ei::Vec2& uv) {
CUDA_FUNCTION bool intersectQuad(const ei::Vec3& p00, const ei::Vec3& p10, const ei::Vec3& p11, const ei::Vec3& p01,
								 const ei::FastRay& ray, float& resT1, float& resT2,
								 ei::Vec2& uv1, ei::Vec2& uv2) {
	// The following equations may degenare if one or two components of the ray.direction
	// are 0. To prevent this we choose the largest component and one other to setup the system.
	// At least one is always greater 0.
	i32 d0 = 0, d1 = 1, d2 = 2;
	if(pabs(ray.direction.x) < pabs(ray.direction.y) ||
		pabs(ray.direction.x) < pabs(ray.direction.z)) {
		if(pabs(ray.direction.y) >= pabs(ray.direction.z)) {
			d0 = 1; d1 = 0; d2 = 2;
		} else {
			d0 = 2; d1 = 1; d2 = 0;
		}
	}

	const ei::Vec3 a = p11 - p10 - p01 + p00;
	const ei::Vec3 b = p10 - p00;
	const ei::Vec3 c = p01 - p00;
	const ei::Vec3 d = p00 - ray.origin;

	const float A1 = a[d1] * ray.direction[d0] - a[d0] * ray.direction[d1];
	const float A2 = a[d2] * ray.direction[d0] - a[d0] * ray.direction[d2];
	const float B1 = b[d1] * ray.direction[d0] - b[d0] * ray.direction[d1];
	const float B2 = b[d2] * ray.direction[d0] - b[d0] * ray.direction[d2];
	const float C1 = c[d1] * ray.direction[d0] - c[d0] * ray.direction[d1];
	const float C2 = c[d2] * ray.direction[d0] - c[d0] * ray.direction[d2];
	const float D1 = d[d1] * ray.direction[d0] - d[d0] * ray.direction[d1];
	const float D2 = d[d2] * ray.direction[d0] - d[d0] * ray.direction[d2];

	// Solve quadratic equ. for number of hitpoints
	float v0 = 0.0f, v1 = 0.0f;
	if(ei::solveSquarePoly(A2*C1 - A1*C2, A2*D1 - A1*D2 + B2*C1 - B1*C2, B2*D1 - B1*D2, v0, v1)) {
		// For the sake of divergence ignore the fact we might only have a single solution
		float u0 = 0.0f, u1 = 0.0f;
		float t0 = -1.f;
		float t1 = -1.f;
		if(v0 >= 0.f && v0 <= 1.f) {
			u0 = computeU(v0, A1, A2, B1, B2, C1, C2, D1, D2);
			if(u0 >= 0.f && u0 <= 1.f) {
				// Use the component with largest ray direction component to avoid singularities
				t0 = (u0*v0 * a[d0] + u0 * b[d0] + v0 * c[d0] + d[d0]) * ray.invDirection[d0];
			}
		}
		if(v1 == v0) {
			// There is only one/no hit point (planar quad)!
			uv1 = ei::Vec2(u0, v0);
			resT1 = t0;
			resT2 = -1.f;
		} else {
			if(v1 >= 0.f && v1 <= 1.f) {
				u1 = computeU(v1, A1, A2, B1, B2, C1, C2, D1, D2);
				if(u1 >= 0.f && u1 <= 1.f) {
					// Use the component with largest ray direction component to avoid singularities
					t1 = (u1*v1 * a[d0] + u1 * b[d0] + v1 * c[d0] + d[d0]) * ray.invDirection[d0];
				}
			}
			if(t0 > 0.f) {
				if(t1 > 0.f && t1 < t0) {
					uv1 = ei::Vec2(u1, v1);
					uv2 = ei::Vec2(u0, v0);
					resT1 = t1;
					resT2 = t0;
				} else {
					uv2 = ei::Vec2(u1, v1);
					uv1 = ei::Vec2(u0, v0);
					resT2 = t1;
					resT1 = t0;
				}
			} else {
				uv1 = ei::Vec2(u1, v1);
				uv2 = ei::Vec2(u0, v0);
				resT1 = t1;
				resT2 = t0;
			}
		}
		return resT1 > 0.f;
	}
	return false;
}

template < Device dev, bool alphatesting >
CUDA_FUNCTION bool intersects_primitve(
	const SceneDescriptor<dev>& scene,
	const LodDescriptor<dev>& obj,
	const ei::FastRay& ray,
	const i32 instanceId,
	const i32 primId,
	float& hitT,				// In out: max hit distance before, if hit then returns the new distance
	SurfaceParametrization& surfParams
) {
	mAssert(primId >= 0);
	if(primId < (i32)obj.polygon.numTriangles) {
		// Triangle.
		const ei::Vec3* meshVertices = obj.polygon.vertices;
		const i32 indexOffset = primId * 3;
		const ei::IVec3 ids = { obj.polygon.vertexIndices[indexOffset],
								obj.polygon.vertexIndices[indexOffset + 1],
								obj.polygon.vertexIndices[indexOffset + 2] };
		const ei::Triangle tri = { meshVertices[ids[0]],
								   meshVertices[ids[1]],
								   meshVertices[ids[2]] };

		float t;
		ei::Vec3 barycentric;
		if(ei::intersects(ray, tri, t, barycentric) && t < hitT && t > 0.0f) {
			// Perform alpha test
			if(alphatesting) {
				MaterialIndex matIdx = obj.polygon.matIndices[primId];
				if(scene.has_alpha(matIdx)) {
					// Compute UV coordinates
					const ei::Vec2 uvV[3] = { obj.polygon.uvs[ids.x], obj.polygon.uvs[ids.y], obj.polygon.uvs[ids.z] };
					const auto uv = uvV[0] * barycentric.x + uvV[1] * barycentric.y +
						uvV[2] * (1.f - barycentric.x - barycentric.y);

					// < 0.5 is the threshold for transparency (binary decision)
					if(textures::sample(scene.get_alpha_texture(matIdx), uv).x < 0.5f)
						return false;
				}
			}

			hitT = t;
			surfParams.barycentric = ei::Vec2{ barycentric.x, barycentric.y };
			return true;
		}
	} else if(primId < (i32)(obj.polygon.numTriangles + obj.polygon.numQuads)) {
		// Quad.
		const i32 indexOffset = (primId - obj.polygon.numTriangles) * 4 + obj.polygon.numTriangles * 3;
		const ei::Vec3* meshVertices = obj.polygon.vertices;
		const u32* ids = obj.polygon.vertexIndices + indexOffset;
		// There are up to two intersections with a quad. Since the closer one
		// could be the self intersection move forward on the ray before testing.
		float t1, t2;
		ei::Vec2 bilin1, bilin2;

		if(intersectQuad(meshVertices[ids[0]], meshVertices[ids[1]], meshVertices[ids[2]], meshVertices[ids[3]], ray,
						 t1, t2, bilin1, bilin2) && t1 < hitT) {
			// Perform alpha test
			if(alphatesting) {
				MaterialIndex matIdx = obj.polygon.matIndices[primId];
				if(scene.has_alpha(matIdx)) {
					// Compute UV coordinates
					const ei::Vec2 uvV[4] = { obj.polygon.uvs[ids[0]], obj.polygon.uvs[ids[1]], obj.polygon.uvs[ids[2]], obj.polygon.uvs[ids[3]] };
					const ei::Vec2 uv1 = ei::bilerp(uvV[0u], uvV[1u], uvV[3u], uvV[2u], bilin1.x, bilin1.y);

					// < 0.5 is the threshold for transparency (binary decision)
					if(textures::sample(scene.get_alpha_texture(matIdx), uv1).x < 0.5f) {
						// Gotta check if we intersect another part of the quad (because it may not be planar)
						if(t2 > 0.f && t2 < hitT) {
							const ei::Vec2 uv2 = ei::bilerp(uvV[0u], uvV[1u], uvV[3u], uvV[2u], bilin2.x, bilin2.y);
							if(textures::sample(scene.get_alpha_texture(matIdx), uv2).x < 0.5f) {
								return false;
							} else {
								t1 = t2;
								bilin1 = bilin2;
							}
						} else {
							return false;
						}
					}
				}
			}

			hitT = t1;
			surfParams.bilinear = bilin1;
			return true;
		}
	} else {
		mAssert(primId < obj.numPrimitives);
		// Sphere.
		const ei::Sphere& sph = obj.spheres.spheres[primId];
		// Because it is important if we start incide or outside it is better
		// to modify the ray beforehand. Testing for tmin afterwards is buggy.
		float t1, t2;
		if(ei::intersects(ray, sph, t1, t2) && t1 < hitT) {
			// Perform alpha test
			if(alphatesting) {
				MaterialIndex matIdx = obj.spheres.matIndices[primId];
				if(scene.has_alpha(matIdx)) {
					// Compute UV coordinates
					const i32 sphId = primId - (obj.polygon.numTriangles + obj.polygon.numQuads);
					const Point center = transform(obj.spheres.spheres[sphId].center, scene.instanceToWorld[instanceId]);
					const ei::Vec3 hitPoint = transform(ray.origin + t1 * ray.direction, scene.instanceToWorld[instanceId]);
					const ei::Vec3 geoNormal = normalize(hitPoint - center); // Normalization required for acos() below
					const ei::Vec3 localN = normalize(transpose(ei::Mat3x3{ scene.instanceToWorld[instanceId] }) * geoNormal);
					const ei::Vec2 uv{
						atan2f(localN.y, localN.x) / (2.0f * ei::PI) + 0.5f,
						acosf(-localN.z) / ei::PI
					};

					// < 0.5 is the threshold for transparency (binary decision)
					if(textures::sample(scene.get_alpha_texture(matIdx), uv).x < 0.5f) {
						// Check if we can see the back side
						if(t2 < hitT && t2 > 0.f) {
							const ei::Vec3 hitPoint2 = transform(ray.origin + t2 * ray.direction, scene.instanceToWorld[instanceId]);
							const ei::Vec3 geoNormal2 = normalize(hitPoint2 - center); // Normalization required for acos() below
							const ei::Vec3 localN2 = normalize(transpose(ei::Mat3x3{ scene.instanceToWorld[instanceId] }) * geoNormal2);
							const ei::Vec2 uv2{
								atan2f(localN2.y, localN2.x) / (2.0f * ei::PI) + 0.5f,
								acosf(-localN2.z) / ei::PI
							};

							if(textures::sample(scene.get_alpha_texture(matIdx), uv2).x < 0.5f)
								return false;
							t1 = t2;
						} else {
							return false;
						}
					}
				}
			}

			hitT = t1;
			// Barycentrics unused; TODO: get coordinates anyway?
			return true;
		}
	}
	return false;
}


// Transition from the top-level BVH to the object space BVH.
// Returns false, if the object is not hit (in which case nothing is changed).
template < Device dev > CUDA_FUNCTION
bool world_to_object_space(const SceneDescriptor<dev>& scene, const i32 instanceId,
						   const ei::FastRay& ray, ei::FastRay& currentRay, float& rayScale,
						   float& tmax, const LodDescriptor<dev>*& obj, const LBVH<dev>*& currentBvh) {
	// Transform the ray into instance-local coordinates
	const ei::Vec3 rayDir = transformDir(ray.direction, scene.worldToInstance[instanceId]);
	float scale = len(rayDir);
	const ei::Ray transRay = { transform(ray.origin, scene.worldToInstance[instanceId]),
							   rayDir / scale };
	ei::FastRay fray { transRay };
	// Scale our current maximum intersection distance into the object space
	// to avoid false negatives.
	const float objSpaceTMax = tmax * scale;

	// Intersect the ray against the object's bounding box.
	float objSpaceT;
	const i32 objId = scene.lodIndices[instanceId];
	const ei::Box& box = scene.aabbs[objId];
	if(intersect(box, fray, objSpaceTMax, objSpaceT)) {
		currentRay = fray;
		rayScale = scale;
		tmax = objSpaceTMax;
		obj = &scene.lods[objId];
		currentBvh = (const LBVH<dev>*)obj->accelStruct.accelParameters;
		return true;
	}
	return false;
}

// Go back to the scene BVH and adjust tmax. Does nothing if the current
// state is already in scene space.
template < Device dev > CUDA_FUNCTION
void object_to_world_space(const ei::FastRay& ray, ei::FastRay& currentRay, float& rayScale,
						   const LBVH<dev>& sceneBvh, const LBVH<dev>*& currentBvh, float& tmax,
						   const LodDescriptor<dev>*& obj) {
	currentRay = ray;
	currentBvh = &sceneBvh;
	tmax /= rayScale;
	rayScale = 1.0f;
	obj = nullptr;
}

} // namespace ::


template < Device dev, bool alphatest > __host__ __device__
RayIntersectionResult first_intersection(
	const SceneDescriptor<dev>& scene,
	ei::Ray& ray,
	const ei::Vec3& geoNormal,
	const float tmax
) {
	add_epsilon(ray.origin, ray.direction, geoNormal);
	// Init scene wide properties
	const LBVH<dev>& bvh = *(const LBVH<dev>*)scene.accelStruct.accelParameters;
	i32 hitPrimId = IGNORE_ID;						// No primitive intersected so far.
	i32 hitInstanceId = IGNORE_ID;
	SurfaceParametrization surfParams;
	const ei::FastRay fray { ray };

	// Set the current traversal state
	float hitT = tmax;						// t-value of the closest intersection.
	ei::FastRay currentRay = fray;
	float currentTScale = 1.0f;
	const LodDescriptor<dev>* obj = nullptr;
	const LBVH<dev>* currentBvh = &bvh;
	i32 currentInstanceId = IGNORE_ID;

	// Setup traversal.
	i32 traversalStack[STACK_SIZE];
	i32 stackIdx = 0;		// Points to the next free stack entry
	i32 nodeAddr = 0;		// Start from the root.
	i32 primCount = 2;		// Internal nodes have two boxes
	i32 primOffset = 0;//TODO: can be removed by simply increasing nodeAddr

	// No Scene-BVH => got to object-space directly
	if(scene.numInstances == 1) {
		if(!world_to_object_space(scene, 0, fray, currentRay, currentTScale, hitT, obj, currentBvh))
			primCount = 0; // No hit of the entire scene, skip the upcoming loop
		currentInstanceId = 0;
		if(obj && obj->numPrimitives == 1)
			primCount = 1;
	}

	// Traversal loop.
	while(stackIdx > 0 || primCount > 0) {
		// Pop top-most entry
		if(primCount == 0) {
			nodeAddr = traversalStack[--stackIdx];
			if(nodeAddr == EntrypointSentinel) { // Was in object BVH, go back to scene
				if(stackIdx == 0) break;
				object_to_world_space(fray, currentRay, currentTScale, bvh, currentBvh, hitT, obj);
				// Pop either the parent nodeAddr or the next instance nodeAddr.
				nodeAddr = traversalStack[--stackIdx];
			}
			if(nodeAddr >= currentBvh->numInternalNodes) { // Leafs additionally store the primitive count
				primCount = traversalStack[--stackIdx];
			} else primCount = 2;
			primOffset = 0;
		}

		while(nodeAddr < currentBvh->numInternalNodes && primCount > 0) { // Internal node?
			// Fetch AABBs of the two child bvh.
			i32 nodeIdx = nodeAddr * 2;
			const BvhNode& left  = currentBvh->bvh[nodeIdx];
			const BvhNode& right = currentBvh->bvh[nodeIdx + 1];

			// Intersect the ray against the children bounds.
			float c0min, c1min;
			bool traverseChild0 = intersect(left.bb, currentRay, hitT, c0min);
			bool traverseChild1 = intersect(right.bb, currentRay, hitT, c1min);

			// Set maximum values to unify upcomming code
			if(!traverseChild0) c0min = 1e38f;
			if(!traverseChild1) c1min = 1e38f;

			// If both children are hit, push the farther one to the stack
			if(traverseChild0 && traverseChild1) {
				i32 pushAddr = (c1min < c0min) ? left.index : right.index;
				if(pushAddr >= currentBvh->numInternalNodes)	// Leaf? Then push the count too
					traversalStack[stackIdx++] = (c1min < c0min) ? left.primCount : right.primCount;
				traversalStack[stackIdx++] = pushAddr;
			}
			// Get address of the closer one (independent of a hit)
			nodeAddr = (c0min <= c1min) ? left.index : right.index;
			if(nodeAddr >= currentBvh->numInternalNodes) {
				primCount = (c0min <= c1min) ? left.primCount : right.primCount;
				primOffset = 0;
			} else primCount = 2;
			// Neither child was intersected => wait for pop stack.
			if(!traverseChild0 && !traverseChild1)
				primCount = 0;
		}

		if(nodeAddr >= currentBvh->numInternalNodes && primCount > 0) { // Leaf?
			const i32 leafId = nodeAddr - currentBvh->numInternalNodes + primOffset;
			const i32 primId = currentBvh->primIds[leafId];
			++primOffset;
			--primCount;

			// Primitves can be instances or true primities.
			if(!obj) {		// Currently in scene BVH => go to object space.
				if(world_to_object_space(scene, primId, fray, currentRay, currentTScale, hitT, obj, currentBvh)) {
					if(obj->numPrimitives == 1) { // Fast path - no BVH
						if(intersects_primitve<dev, alphatest>(scene, *obj, currentRay, primId, 0, hitT, surfParams)) {
							hitInstanceId = primId;
							hitPrimId = 0;
						}
						// Immediatelly leave the object space again
						object_to_world_space(fray, currentRay, currentTScale, bvh, currentBvh, hitT, obj);
					} else {
						// Push a marker and the remaining number of instances to the stack.
						// This is necessary to know when we must use the backward transition.
						if(primCount > 0) {
							traversalStack[stackIdx++] = primCount;
							traversalStack[stackIdx++] = nodeAddr + primOffset;
						}
						traversalStack[stackIdx++] = EntrypointSentinel;
						// Clean restart in local BVH
						nodeAddr = 0;
						primCount = 2;
						primOffset = 0;
						currentInstanceId = primId;
					}
				}
			} else if(intersects_primitve<dev, alphatest>(scene, *obj, currentRay, currentInstanceId, primId, hitT, surfParams)) {
				hitInstanceId = currentInstanceId;
				hitPrimId = primId;
			}
		}
	}

	if(obj)
		object_to_world_space(fray, currentRay, currentTScale, bvh, currentBvh, hitT, obj);

	// Nobody should update hitT if no primitive is hit
	mAssert((hitInstanceId != IGNORE_ID && hitPrimId != IGNORE_ID) || ei::approx(hitT, tmax));

	/* TEST CODE WHICH MAKES A LINEAR TEST (without the BVH)
	for(int i = 0; i < scene.numInstances; ++i) {
		auto& obj = scene.lods[ scene.lodIndices[i] ];
		ei::Ray transRay = { transform(ray.origin, scene.worldToInstance[i]),
							 normalize(transformDir(ray.direction, scene.worldToInstance[i])) };
		for(int p = 0; p < obj.numPrimitives; ++p) {
			if(intersects_primitve(scene, obj, transRay, p, -1, i, hitPrimId, hitT, surfParams))
				hitInstanceId = i;
		}
	}*/

	if(hitInstanceId == IGNORE_ID) {
		return { hitT, { IGNORE_ID, IGNORE_ID } };
	} else {
		// To be determined
		ei::Vec3 geoNormal;
		ei::Vec3 tangentX;
		ei::Vec3 tangentY;
		ei::Vec2 uv;

		i32 primId = hitPrimId;

		const LodDescriptor<dev>& obj = scene.lods[scene.lodIndices[hitInstanceId]];

		const i32 offsetSpheres = obj.polygon.numTriangles + obj.polygon.numQuads;
		if(primId >= offsetSpheres) { // Sphere?
			const i32 sphId = primId - offsetSpheres;
			const ei::Vec3 hitPoint = ray.origin + hitT * ray.direction;
			const Point center = transform(obj.spheres.spheres[sphId].center, scene.instanceToWorld[hitInstanceId]);
			geoNormal = normalize(hitPoint - center); // Normalization required for acos() below

			if(geoNormal.x == 0.0f && geoNormal.y == 0.0f)
				tangentX = ei::Vec3(1.0f, 0.0f, 0.0f);
			else
				tangentX = ei::Vec3(normalize(ei::Vec2(geoNormal.y, -geoNormal.x)), 0.0f);

			const ei::Vec3 localN = normalize(transpose(ei::Mat3x3{ scene.instanceToWorld[hitInstanceId] }) * geoNormal);
			uv.x = atan2f(localN.y, localN.x) / (2.0f * ei::PI) + 0.5f;
			uv.y = acosf(-localN.z) / ei::PI;
			surfParams.st = uv;
			return RayIntersectionResult{ hitT, { hitInstanceId, hitPrimId }, geoNormal, tangentX, uv, surfParams };
		} else {
			const i32* indices = (i32*)obj.polygon.vertexIndices;
			const ei::Vec3* meshVertices = obj.polygon.vertices;
			const ei::Vec2* meshUVs = obj.polygon.uvs;
			if(primId < (i32)obj.polygon.numTriangles) {
				// Triangle.
				u32 triId = primId * 3;
				ei::IVec3 ids = { indices[triId],
								  indices[triId + 1],
								  indices[triId + 2] };
				const ei::Vec3 v[3] = { meshVertices[ids.x], meshVertices[ids.y], meshVertices[ids.z] };
				const ei::Vec2 uvV[3] = { meshUVs[ids.x], meshUVs[ids.y], meshUVs[ids.z] };
				// Compute the tangent space by solving LES
				const ei::Vec3 dx0 = v[1u] - v[0u];
				const ei::Vec3 dx1 = v[2u] - v[0u];
				const ei::Vec2 du0 = uvV[1u] - uvV[0u];
				const ei::Vec2 du1 = uvV[2u] - uvV[0u];
				float det = (du0.x * du1.y - du0.y * du1.x);
				// TODO: fetch the instance instead (issue #44)
				if(det >= 1e-5f || det <= -1e5f)
					tangentX = (dx0 * du1.y - dx1 * du0.y) / det;
				else tangentX = dx0;

				// Don't use the UV tangents to compute the normal, since they may be reversed
				geoNormal = cross(dx0, dx1);

				mAssert(dot(geoNormal, obj.polygon.normals[ids.x]) > 0.f);

				uv = uvV[0] * surfParams.barycentric.x + uvV[1] * surfParams.barycentric.y +
					uvV[2] * (1.f - surfParams.barycentric.x - surfParams.barycentric.y);
			} else {
				// Quad.
				i32 quadId = (primId - obj.polygon.numTriangles) * 4;
				indices += obj.polygon.numTriangles * 3;
				ei::IVec4 ids = { indices[quadId + 0],
								  indices[quadId + 1],
								  indices[quadId + 2],
								  indices[quadId + 3] };
				const ei::Vec3 v[4] = { meshVertices[ids.x], meshVertices[ids.y], meshVertices[ids.z], meshVertices[ids.w] };
				const ei::Vec2 uvV[4] = { meshUVs[ids.x], meshUVs[ids.y], meshUVs[ids.z], meshUVs[ids.w] };
				// Compute tangent space by using surrogate coordinate system to get interpolated UVs
				// https://stackoverflow.com/questions/47187600/differences-in-calculating-tbn-matrix-for-triangles-versus-quads
				// TODO: fetch the instance instead (issue #44)
				const ei::Vec3 dxds = (1.f - surfParams.bilinear.u) * (v[3u] - v[0u]) + surfParams.bilinear.u * (v[2u] - v[1u]);
				const ei::Vec3 dxdt = (1.f - surfParams.bilinear.v) * (v[1u] - v[0u]) + surfParams.bilinear.v * (v[2u] - v[3u]);
				const ei::Matrix<float, 3, 2> dxdst{
					dxds.x, dxdt.x,
					dxds.y, dxdt.y,
					dxds.z, dxdt.z
				};
				const ei::Vec2 duds = (1.f - surfParams.bilinear.u) * (uvV[3u] - uvV[0u]) + surfParams.bilinear.u * (uvV[2u] - uvV[1u]);
				const ei::Vec2 dudt = (1.f - surfParams.bilinear.v) * (uvV[1u] - uvV[0u]) + surfParams.bilinear.v * (uvV[2u] - uvV[3u]);
				const ei::Matrix<float, 2, 2> dudst{
					duds.x, dudt.x,
					duds.y, dudt.y,
				};
				float detDudst = dudst[0] * dudst[3] - dudst[1] * dudst[2];
				if(detDudst >= 1e-5f || detDudst <= -1e5f)
					tangentX = dxdst * ei::Vec2 { dudst[3] / detDudst, -dudst[2] / detDudst };
				else tangentX = dxds;
				//const ei::Mat2x2 dsduv = ei::invert(dudst);
				//const ei::Matrix<float, 3, 2> tangents = dxdst * dsduv;
				//tangentX = ei::Vec3{ tangents(0, 0), tangents(1, 0), tangents(2, 0) };
				//tangentY = ei::Vec3{ tangents(0, 1), tangents(1, 1), tangents(2, 1) };

				geoNormal = cross(dxdt, dxds);
				uv = ei::bilerp(uvV[0u], uvV[1u], uvV[3u], uvV[2u], surfParams.bilinear.x, surfParams.bilinear.y);
			}
		}

		// Transform the normal and tangents into world space
		// Polygon objects are allowed to have a non-uniform scaling
		ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[hitInstanceId] });
		geoNormal = normalize(rotationInvScale * geoNormal);
		tangentX = normalize(rotationInvScale * tangentX);

		mAssert(!(isnan(tangentX.x) || isnan(tangentX.y) || isnan(tangentX.z)));
		mAssert(!(isnan(geoNormal.x) || isnan(geoNormal.y) || isnan(geoNormal.z)));

		return RayIntersectionResult{ hitT, { hitInstanceId, hitPrimId }, geoNormal, tangentX, uv, surfParams };
	}
}


template < Device dev > __host__ __device__
bool any_intersection(
	const SceneDescriptor<dev>& scene,
	scene::Point a,
	scene::Point b,
	const scene::Direction& geoNormalA,
	const scene::Direction& geoNormalB,
	const scene::Direction& connectionDirAtoB
) {
	add_epsilon(a,  connectionDirAtoB, geoNormalA);
	add_epsilon(b, -connectionDirAtoB, geoNormalB);
	ei::Vec3 correctedDir = b - a;
	float tmax = len(correctedDir);
	const ei::Ray ray { a, correctedDir / tmax };
	// Init scene wide properties
	const LBVH<dev>& bvh = *(const LBVH<dev>*)scene.accelStruct.accelParameters;
	const ei::FastRay fray { ray };

	// Set the current traversal state
	ei::FastRay currentRay = fray;
	float currentTScale = 1.0f;
	const LodDescriptor<dev>* obj = nullptr;
	const LBVH<dev>* currentBvh = &bvh;
	i32 currentInstanceId = IGNORE_ID;

	// Setup traversal.
	i32 traversalStack[STACK_SIZE];
	i32 stackIdx = 0;		// Points to the next free stack entry
	i32 nodeAddr = 0;		// Start from the root.
	i32 primCount = 2;		// Internal nodes have two boxes
	i32 primOffset = 0;//TODO: can be removed by simply increasing nodeAddr

	// No Scene-BVH => got to object-space directly
	if(scene.numInstances == 1) {
		if(!world_to_object_space(scene, 0, fray, currentRay, currentTScale, tmax, obj, currentBvh))
			return false; // No hit of the entire scene
		currentInstanceId = 0;
		if(obj != nullptr && obj->numPrimitives == 1)
			primCount = 1;
	}

	// Traversal loop.
	while(stackIdx > 0 || primCount > 0) {
		// Pop top-most entry
		if(primCount == 0) {
			nodeAddr = traversalStack[--stackIdx];
			if(nodeAddr == EntrypointSentinel) { // Was in object BVH, go back to scene
				if(stackIdx == 0) break;
				object_to_world_space(fray, currentRay, currentTScale, bvh, currentBvh, tmax, obj);
				// Pop either the parent nodeAddr or the next instance nodeAddr.
				nodeAddr = traversalStack[--stackIdx];
			}
			if(nodeAddr >= currentBvh->numInternalNodes) { // Leafs additionally store the primitive count
				primCount = traversalStack[--stackIdx];
			} else primCount = 2;
			primOffset = 0;
		}

		while(nodeAddr < currentBvh->numInternalNodes && primCount > 0) { // Internal node?
			// Fetch AABBs of the two child bvh.
			i32 nodeIdx = nodeAddr * 2;
			const BvhNode& left  = currentBvh->bvh[nodeIdx];
			const BvhNode& right = currentBvh->bvh[nodeIdx + 1];

			// Intersect the ray against the children bounds.
			float c0min, c1min;
			bool traverseChild0 = intersect(left.bb, currentRay, tmax, c0min);
			bool traverseChild1 = intersect(right.bb, currentRay, tmax, c1min);

			// Set maximum values to unify upcomming code
			if(!traverseChild0) c0min = 1e38f;
			if(!traverseChild1) c1min = 1e38f;

			// If both children are hit, push the farther one to the stack
			if(traverseChild0 && traverseChild1) {
				i32 pushAddr = (c1min < c0min) ? left.index : right.index;
				if(pushAddr >= currentBvh->numInternalNodes)	// Leaf? Then push the count too
					traversalStack[stackIdx++] = (c1min < c0min) ? left.primCount : right.primCount;
				traversalStack[stackIdx++] = pushAddr;
			}
			// Get address of the closer one (independent of a hit)
			nodeAddr = (c0min <= c1min) ? left.index : right.index;
			if(nodeAddr >= currentBvh->numInternalNodes) {
				primCount = (c0min <= c1min) ? left.primCount : right.primCount;
				primOffset = 0;
			} else primCount = 2;
			// Neither child was intersected => wait for pop stack.
			if(!traverseChild0 && !traverseChild1)
				primCount = 0;
		}

		if(nodeAddr >= currentBvh->numInternalNodes && primCount > 0) { // Leaf?
			const i32 leafId = nodeAddr - currentBvh->numInternalNodes + primOffset;
			const i32 primId = currentBvh->primIds[leafId];
			++primOffset;
			--primCount;

			SurfaceParametrization surfParams;

			// Primitves can be instances or true primities.
			if(!obj) {		// Currently in scene BVH => go to object space.
				if(world_to_object_space(scene, primId, fray, currentRay, currentTScale, tmax, obj, currentBvh)) {
					if(obj->numPrimitives == 1) { // Fast path - no BVH
						if(intersects_primitve<dev, true>(scene, *obj, currentRay, primId, 0, tmax, surfParams)) {
							return true;
						}
						// Immediatelly leave the object space again
						object_to_world_space(fray, currentRay, currentTScale, bvh, currentBvh, tmax, obj);
					} else {
						// Push a marker and the remaining number of instances to the stack.
						// This is necessary to know when we must use the backward transition.
						if(primCount > 0) {
							traversalStack[stackIdx++] = primCount;
							traversalStack[stackIdx++] = nodeAddr + primOffset;
						}
						traversalStack[stackIdx++] = EntrypointSentinel;
						// Clean restart in local BVH
						nodeAddr = 0;
						primCount = 2;
						primOffset = 0;
						currentInstanceId = primId;
					}
				}
			} else if(intersects_primitve<dev, true>(scene, *obj, currentRay, currentInstanceId, primId, tmax, surfParams)) {
				return true;
			}
		}
	}
	return false;
}


template __host__ __device__ bool any_intersection(
	const SceneDescriptor<Device::CUDA>&,
	scene::Point, scene::Point, const scene::Direction&, const scene::Direction&,
	const scene::Direction&
);

template __host__ __device__ bool any_intersection(
	const SceneDescriptor<Device::CPU>&,
	scene::Point, scene::Point, const scene::Direction&, const scene::Direction&,
	const scene::Direction&
);

template __host__ __device__ RayIntersectionResult first_intersection<Device::CUDA, true>(
	const SceneDescriptor<Device::CUDA>&, ei::Ray&, 
	const ei::Vec3&, const float
);
template __host__ __device__ RayIntersectionResult first_intersection<Device::CUDA, false>(
	const SceneDescriptor<Device::CUDA>&, ei::Ray&,
	const ei::Vec3&, const float
	);

template __host__ __device__ RayIntersectionResult first_intersection<Device::CPU, true>(
	const SceneDescriptor<Device::CPU>&, ei::Ray&,
	const ei::Vec3&, const float
);
template __host__ __device__ RayIntersectionResult first_intersection<Device::CPU, false>(
	const SceneDescriptor<Device::CPU>&, ei::Ray&,
	const ei::Vec3&, const float
	);

}}} // namespace mufflon::scene::accel_struct
