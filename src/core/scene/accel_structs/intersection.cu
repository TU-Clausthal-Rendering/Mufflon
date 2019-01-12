#include "intersection.hpp"
#include "accel_structs_commons.hpp"
#include "lbvh.hpp"
#include "util/types.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <ei/3dintersection.hpp>

namespace mufflon { namespace scene { namespace accel_struct {

namespace {

#define STACK_SIZE              96 //64          // Size of the traversal stack in local memory.
#define OBJ_STACK_SIZE              64 //64          // Size of the traversal stack in local memory.
enum : i32 {
	EntrypointSentinel = (i32)0xFFFFFFFF,   // Bottom-most stack entry, indicating the end of traversal.
	IGNORE_ID = (i32)0xFFFFFFFF
};

// Experimentally determined best mix of float/i32/video minmax instructions for Kepler.
__device__ __inline__ i32   min_min(i32 a, i32 b, i32 c) { i32 v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ i32   min_max(i32 a, i32 b, i32 c) { i32 v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ i32   max_min(i32 a, i32 b, i32 c) { i32 v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ i32   max_max(i32 a, i32 b, i32 c) { i32 v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }


CUDA_FUNCTION bool intersect(const ei::Vec3& boxMin, const ei::Vec3& boxMax,
	const ei::Vec3 invDir, const ei::Vec3 ood, 
	const float tmin, const float tmax, float& cmin) {//, float& cmax) {
#ifdef __CUDA_ARCH__
	ei::Vec3 lo = boxMin * invDir - ood;
	ei::Vec3 hi = boxMax * invDir - ood;
	cmin = spanBeginKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
	const float cmax = spanEndKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);
	return cmin <= cmax;
#else
	// TODO: use the epsilon one? FastRay one?
	float t0 = boxMin.x * invDir.x - ood.x;
	float t1 = boxMax.x * invDir.x - ood.x;
	cmin = ei::min(t0, t1);
	float cmax = ei::max(t0, t1);
	if (cmax < tmin || cmin > tmax) return false;
	t0 = boxMin.y * invDir.y - ood.y;
	t1 = boxMax.y * invDir.y - ood.y;
	float min2 = ei::min(t0, t1);
	float max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	if (cmax < tmin || cmin > tmax || cmin > cmax) return false;
	t0 = boxMin.z * invDir.z - ood.z;
	t1 = boxMax.z * invDir.z - ood.z;
	min2 = ei::min(t0, t1);
	max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	return (cmax >= tmin) && (cmin <= tmax) && (cmin <= cmax);
#endif // __CUDA_ARCH__
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
CUDA_FUNCTION float intersectQuad(const ei::Tetrahedron& quad, const ei::Ray& ray, ei::Vec2& uv) {
	// Implementation from http://www.sci.utah.edu/~kpotter/publications/ramsey-2004-RBPI.pdf
	const ei::Vec3& p00 = quad.v0;
	const ei::Vec3& p10 = quad.v1;
	const ei::Vec3& p01 = quad.v3;
	const ei::Vec3& p11 = quad.v2;

	const ei::Vec3 a = p11 - p10 - p01 + p00;
	const ei::Vec3 b = p10 - p00;
	const ei::Vec3 c = p01 - p00;
	const ei::Vec3 d = p00;

	const float AXY = a.y*ray.direction.x - a.x*ray.direction.y;
	const float AXZ = a.z*ray.direction.x - a.x*ray.direction.z;
	const float AYZ = a.z*ray.direction.y - a.y*ray.direction.z;
	const float BXY = b.y*ray.direction.x - b.x*ray.direction.y;
	const float BXZ = b.z*ray.direction.x - b.x*ray.direction.z;
	const float BYZ = b.z*ray.direction.y - b.y*ray.direction.z;
	const float CXY = c.y*ray.direction.x - c.x*ray.direction.y;
	const float CXZ = c.z*ray.direction.x - c.x*ray.direction.z;
	const float CYZ = c.z*ray.direction.y - c.y*ray.direction.z;
	const float DXY = (d.y - ray.origin.y) * ray.direction.x - (d.x - ray.origin.x) * ray.direction.y;
	const float DXZ = (d.z - ray.origin.z) * ray.direction.x - (d.x - ray.origin.x) * ray.direction.z;
	const float DYZ = (d.z - ray.origin.z) * ray.direction.y - (d.y - ray.origin.y) * ray.direction.z;

	float A1, A2, B1, B2, C1, C2, D1, D2;
		// Use the component with largest ray direction component to avoid singularities
	if(ei::abs(ray.direction.x) >= ei::abs(ray.direction.y) && ei::abs(ray.direction.x) >= ei::abs(ray.direction.z)) {
		A1 = AXY;
		B1 = BXY;
		C1 = CXY;
		D1 = DXY;
		A2 = AXZ;
		B2 = BXZ;
		C2 = CXZ;
		D2 = DXZ;
	} else if(ei::abs(ray.direction.y) >= ei::abs(ray.direction.z)) {
		A1 = -AXY;
		B1 = -BXY;
		C1 = -CXY;
		D1 = -DXY;
		A2 = AYZ;
		B2 = BYZ;
		C2 = CYZ;
		D2 = DYZ;
	} else {
		A1 = AXZ;
		B1 = BXZ;
		C1 = CXZ;
		D1 = DXZ;
		A2 = AYZ;
		B2 = BYZ;
		C2 = CYZ;
		D2 = DYZ;
	}

	// Solve quadratic equ. for number of hitpoints
	float t = -1.f;
	float v0, v1;
	if(ei::solveSquarePoly(A2*C1 - A1*C2, A2*D1 - A1*D2 + B2*C1 - B1*C2, B2*D1 - B1*D2, v0, v1)) {
		// For the sake of divergence ignore the fact we might only have a single solution
		float u0, u1;
		float t0 = -1.f;
		float t1 = -1.f;
		if(v0 >= 0.f && v0 <= 1.f) {
			u0 = computeU(v0, A1, A2, B1, B2, C1, C2, D1, D2);
			if(u0 >= 0.f && u0 <= 1.f) {
				if(ei::abs(ray.direction.x) >= ei::abs(ray.direction.y) &&
				   ei::abs(ray.direction.x) >= ei::abs(ray.direction.z))
					t0 = (u0*v0*a.x + u0 * b.x + v0 * c.x + d.x - ray.origin.x) / ray.direction.x;
				else if(ei::abs(ray.direction.y) >= ei::abs(ray.direction.z))
					t0 = (u0*v0*a.y + u0 * b.y + v0 * c.y + d.y - ray.origin.y) / ray.direction.y;
				else
					t0 = (u0*v0*a.z + u0 * b.z + v0 * c.z + d.z - ray.origin.z) / ray.direction.z;
			}
		}
		if(v1 >= 0.f && v1 <= 1.f) {
			u1 = computeU(v1, A1, A2, B1, B2, C1, C2, D1, D2);
			if(u1 >= 0.f && u1 <= 1.f) {
				if(ei::abs(ray.direction.x) >= ei::abs(ray.direction.y) &&
				   ei::abs(ray.direction.x) >= ei::abs(ray.direction.z))
					t1 = (u1*v1*a.x + u1 * b.x + v1 * c.x + d.x - ray.origin.x) / ray.direction.x;
				else if(ei::abs(ray.direction.y) >= ei::abs(ray.direction.z))
					t1 = (u1*v1*a.y + u1 * b.y + v1 * c.y + d.y - ray.origin.y) / ray.direction.y;
				else
					t1 = (u1*v1*a.z + u1 * b.z + v1 * c.z + d.z - ray.origin.z) / ray.direction.z;
			}
		}
		if(t0 > 0.f) {
			if(t1 > 0.f && t1 < t0) {
				uv = ei::Vec2(u1, v1);
				t = t1;
			} else {
				uv = ei::Vec2(u0, v0);
				t = t0;
			}
		} else {
			uv = ei::Vec2(u1, v1);
			t = t1;
		}
	}

	return t;
}

template < Device dev >
CUDA_FUNCTION bool intersects_primitve(
	const ObjectDescriptor<dev>& obj,
	const ei::Ray& ray,
	const i32 primId,
	const i32 startPrimId,
	int& hitPrimId,
	float& hitT,				// In out: max hit distance before, if hit then returns the new distance
	SurfaceParametrization& surfParams
) {
	if(primId < (i32)obj.polygon.numTriangles) {
		// Triangle.
		if(startPrimId == primId) return false; // Masking to avoid self intersections

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
		if(ei::intersects(ray, tri, t, barycentric) && t < hitT) {
			hitT = t;
			surfParams.barycentric = ei::Vec2{ barycentric.x, barycentric.y };
			hitPrimId = primId;
			return true;
		}
	} else if(primId < (i32)(obj.polygon.numTriangles + obj.polygon.numQuads)) {
		// Quad.
		const i32 indexOffset = (primId - obj.polygon.numTriangles) * 4 + obj.polygon.numTriangles * 3;
		const ei::Vec3* meshVertices = obj.polygon.vertices;

		// Check first triangle
		if(startPrimId != primId) { // Masking to avoid self intersections
			const ei::IVec4 ids = { obj.polygon.vertexIndices[indexOffset],
									obj.polygon.vertexIndices[indexOffset + 1],
									obj.polygon.vertexIndices[indexOffset + 2],
									obj.polygon.vertexIndices[indexOffset + 3] };
			const ei::Tetrahedron quad = { meshVertices[ids[0]],
										   meshVertices[ids[1]],
										   meshVertices[ids[2]],
										   meshVertices[ids[3]] };
			ei::Vec2 bilinear;
			const float t = intersectQuad(quad, ray, bilinear);

			if(t > 0.f && t < hitT) {
				hitT = t;
				surfParams.bilinear = bilinear;
				hitPrimId = primId;
				return true;
			}
		}
	} else {
		// Sphere.
		if(startPrimId == primId) return false; // Masking to avoid self intersections
		const ei::Sphere& sph = obj.spheres.spheres[primId];
		float t;
		// TODO: use some epsilon?
		if(ei::intersects(ray, sph, t) && t < hitT) {
			hitT = t;
			hitPrimId = primId;
			// Barycentrics unused; TODO: get coordinates anyway?
			return true;
		}
	}
	return false;
}

} // namespace ::

template < Device dev >
CUDA_FUNCTION bool any_intersection_obj_lbvh_imp(
	const LBVH& bvh,
	const ObjectDescriptor<dev>& obj,
	const ei::Ray& ray,
	const i32 startPrimId,
	const ei::Vec3& invDir, 
	const ei::Vec3& ood,
	const float tmin,
	const float tmax,
	i32* traversalStack
) {
	// Since all threads go to the following branch if numPrimitives == 1,
	// there is no problem with branching.
	if(obj.numPrimitives == 1) {
		float hitT = tmax;
		SurfaceParametrization surfParams;
		i32 hitPrimitiveId;
		if(intersects_primitve(obj, ray, 0, startPrimId, hitPrimitiveId, hitT, surfParams)) {
			return true;
		}
		return false;
	}

	// Setup traversal.
	traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.
	i32 nodeAddr = 0; // Start from the root.  
	i32* stackPtr = traversalStack; // Current position in traversal stack.
	i32 primCount = 0; // Internal nodes have no primitives

	// Traversal loop.
	while(nodeAddr != EntrypointSentinel) {
		if(nodeAddr < bvh.numInternalNodes) { // Internal node?
			// Fetch AABBs of the two child bvh.
			i32 nodeIdx = nodeAddr * 4;
			const ei::Vec4 Lmin_cL = bvh.bvh[nodeIdx];
			const ei::Vec4 Lmax_nL = bvh.bvh[nodeIdx + 1];
			const ei::Vec4 Rmin_cR = bvh.bvh[nodeIdx + 2];
			const ei::Vec4 Rmax_nR = bvh.bvh[nodeIdx + 3];

			// Intersect the ray against the children bounds.
			float c0min, c1min;
			bool traverseChild0 = intersect(ei::Vec3{Lmin_cL}, ei::Vec3{Lmax_nL}, invDir, ood, tmin, tmax, c0min);
			bool traverseChild1 = intersect(ei::Vec3{Rmin_cR}, ei::Vec3{Rmax_nR}, invDir, ood, tmin, tmax, c1min);

			// Neither child was intersected => pop stack.
			if(!traverseChild0 && !traverseChild1) {
				nodeAddr = *stackPtr;
				--stackPtr;
				if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
					primCount = *stackPtr;
					--stackPtr;
				}
			}
			// Otherwise => fetch child pointers.
			else {
				nodeAddr = traverseChild0 ? float_bits_as_int(Lmin_cL.w) : float_bits_as_int(Rmin_cR.w);
				primCount = traverseChild0 ? float_bits_as_int(Lmax_nL.w) : float_bits_as_int(Rmax_nR.w);

				// Both children were intersected => push the farther one.
				if (traverseChild0 && traverseChild1) {
					i32 pushAddr = float_bits_as_int(Rmin_cR.w); // nodeAddr is Lmin_cL.w, this is the other one
					i32 pushCount = float_bits_as_int(Rmax_nR.w);
					if (c1min < c0min) {
						i32 tmp = nodeAddr;
						nodeAddr = pushAddr;
						pushAddr = tmp;
						tmp = primCount;
						primCount = pushCount;
						pushCount = tmp;
					}
					if(pushAddr >= bvh.numInternalNodes) { // Leaf? Then push the count too
						++stackPtr;
						*stackPtr = pushCount;
					}
					++stackPtr;
					*stackPtr = pushAddr;
				}
			}
		}

		if(nodeAddr >= bvh.numInternalNodes && nodeAddr != EntrypointSentinel) { // Leaf?
			const i32 primId = nodeAddr - bvh.numInternalNodes;

			// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
			for(i32 i = 0; i < primCount; i++) {
				float hitT = tmax;
				SurfaceParametrization surfParams;
				i32 hitPrimitiveId;
				if(intersects_primitve(obj, ray, bvh.primIds[primId + i], startPrimId, hitPrimitiveId, hitT, surfParams))
					return true;
			}

			// Pop next node.
			nodeAddr = *stackPtr;
			--stackPtr;
			if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
				primCount = *stackPtr;
				--stackPtr;
			}
		}
	}
	return false;
}


template < Device dev >
CUDA_FUNCTION bool first_intersection_obj_lbvh_imp(
	const LBVH& bvh,
	const ObjectDescriptor<dev>& obj,
	const ei::Ray& ray,
	const i32 startPrimId,
	const ei::Vec3& invDir, 
	const ei::Vec3& ood,
	const float tmin,
	int& hitPrimId, float& hitT,
	SurfaceParametrization& surfParams,
	i32* traversalStack
) {
	// Fast path - no BVH
	if(obj.numPrimitives == 1) {
		return intersects_primitve(obj, ray, 0, startPrimId,
			hitPrimId, hitT, surfParams);
	}
	
	// Setup traversal.
	traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.
	i32 nodeAddr = 0; // Start from the root.  
	i32* stackPtr = traversalStack; // Current position in traversal stack.
	i32 primCount = 0; // Internal nodes have no primitives

	bool hasHit = false;

	// Traversal loop.
	while(nodeAddr != EntrypointSentinel) {
		if(nodeAddr < bvh.numInternalNodes) { // Internal node?
			// Fetch AABBs of the two child bvh.
			i32 nodeIdx = nodeAddr * 4;
			const ei::Vec4 Lmin_cL = bvh.bvh[nodeIdx];
			const ei::Vec4 Lmax_nL = bvh.bvh[nodeIdx + 1];
			const ei::Vec4 Rmin_cR = bvh.bvh[nodeIdx + 2];
			const ei::Vec4 Rmax_nR = bvh.bvh[nodeIdx + 3];

			// Intersect the ray against the children bounds.
			float c0min, c1min;
			bool traverseChild0 = intersect(ei::Vec3{Lmin_cL}, ei::Vec3{Lmax_nL}, invDir, ood, tmin, hitT, c0min);
			bool traverseChild1 = intersect(ei::Vec3{Rmin_cR}, ei::Vec3{Rmax_nR}, invDir, ood, tmin, hitT, c1min);

			// Neither child was intersected => pop stack.
			if(!traverseChild0 && !traverseChild1) {
				nodeAddr = *stackPtr;
				--stackPtr;
				if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
					primCount = *stackPtr;
					--stackPtr;
				}
			}
			// Otherwise => fetch child pointers.
			else {
				nodeAddr = traverseChild0 ? float_bits_as_int(Lmin_cL.w) : float_bits_as_int(Rmin_cR.w);
				primCount = traverseChild0 ? float_bits_as_int(Lmax_nL.w) : float_bits_as_int(Rmax_nR.w);

				// Both children were intersected => push the farther one.
				if (traverseChild0 && traverseChild1) {
					i32 pushAddr = float_bits_as_int(Rmin_cR.w); // nodeAddr is Lmin_cL.w, this is the other one
					i32 pushCount = float_bits_as_int(Rmax_nR.w);
					if (c1min < c0min) {
						i32 tmp = nodeAddr;
						nodeAddr = pushAddr;
						pushAddr = tmp;
						tmp = primCount;
						primCount = pushCount;
						pushCount = tmp;
					}
					if(pushAddr >= bvh.numInternalNodes) { // Leaf? Then push the count too
						++stackPtr;
						*stackPtr = pushCount;
					}
					++stackPtr;
					*stackPtr = pushAddr;
				}
			}
		}

		if(nodeAddr >= bvh.numInternalNodes && nodeAddr != EntrypointSentinel) { // Leaf?
			const i32 primId = nodeAddr - bvh.numInternalNodes;

			// All intersection distances are in this instance's object space
			// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
			for(i32 i = 0; i < primCount; i++) {
				if(intersects_primitve(obj, ray, bvh.primIds[primId+i], startPrimId, hitPrimId, hitT, surfParams))
					hasHit = true;
			}

			// Pop next node.
			nodeAddr = *stackPtr;
			--stackPtr;
			if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
				primCount = *stackPtr;
				--stackPtr;
			}
		}
	}
	return hasHit;
}


template < Device dev > CUDA_FUNCTION
void first_intersection_scene_obj_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const i32 instanceId,
	i32* traversalStack,
	float& hitT,
	i32& hitInstanceId,
	i32& hitPrimId,
	SurfaceParametrization& surfParams
) {
	const ei::Mat3x3 rotScale = ei::Mat3x3{ scene.transformations[instanceId] };
	const ei::Mat3x3 invRotScale = ei::invert(rotScale);
	const ei::Vec3 invTranslation { -scene.transformations[instanceId][3],
									-scene.transformations[instanceId][7],
									-scene.transformations[instanceId][11] };
	const ei::Ray transRay = { invRotScale * (ray.origin + invTranslation),
						 normalize(invRotScale * ray.direction) };
	const ei::Vec3 invDir = sdiv(1.0f, transRay.direction);
	const ei::Vec3 ood = transRay.origin * invDir;

	const i32 objId = scene.objectIndices[instanceId];
	const ei::Box& box = scene.aabbs[objId];
	const float tmin = 1e-6f * len(box.max - box.min);

	// Determine the scale of the transformation to scale the intersection distance back into world space
	// This is only valid for uniform scaling!
	const float transScale = cbrtf(ei::determinant(rotScale));
	// Scale our current maximum intersection distance into the object space to avoid false negatives
	float objSpaceHitT = hitT / transScale;
	const float objSpaceMinT = tmin / transScale;

	// Intersect the ray against the obj bounding box.
	float objSpaceT;
	if(intersect(box.min, box.max, invDir, ood, objSpaceMinT, objSpaceHitT, objSpaceT)) {
		// Intersect the ray against the obj primitive bvh.
		const ObjectDescriptor<dev>& obj = scene.objects[objId];
		const LBVH* lbvh = (LBVH*)obj.accelStruct.accelParameters;
		const i32 checkPrimId = (startInsPrimId.instanceId == instanceId) ? startInsPrimId.primId : IGNORE_ID;
		if (first_intersection_obj_lbvh_imp(
			*lbvh, obj, transRay, checkPrimId, invDir, ood, objSpaceMinT,
			hitPrimId, objSpaceHitT, surfParams, traversalStack)) {
			// Translate the object-space distance into world space again
			hitT = objSpaceHitT * transScale;
			hitInstanceId = instanceId;
		}
	}
}

template < Device dev > __host__ __device__
RayIntersectionResult first_intersection_scene_lbvh_imp(
	const LBVH& bvh,
	const SceneDescriptor<dev>& scene,
	const ei::Ray ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
) {
	const float tmin = 1e-7f * len(scene.aabb.max - scene.aabb.min);
	i32 hitPrimId = IGNORE_ID;						// No primitive intersected so far.
	i32 hitInstanceId = IGNORE_ID;
	SurfaceParametrization surfParams;
	float hitT = tmax;						// t-value of the closest intersection.

	if(scene.numInstances == 1) {
		i32 traversalStack[OBJ_STACK_SIZE];
		first_intersection_scene_obj_lbvh(
			scene, ray, startInsPrimId, 0, traversalStack,
			hitT, hitInstanceId, hitPrimId, surfParams);
	} else {
		const ei::Vec3 invDir = sdiv(1.0f, ray.direction);
		const ei::Vec3 ood = ray.origin * invDir;

		// Setup traversal.
		// Traversal stack in CUDA thread-local memory.
		i32 traversalStack[STACK_SIZE];
		traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.
		i32 nodeAddr = 0; // Start from the root.
		i32 primCount = 0; // Internal nodes have no primitives
		i32* stackPtr = traversalStack; // Current position in traversal stack.

		// Traversal loop.
		while(nodeAddr != EntrypointSentinel) {
			if(nodeAddr < bvh.numInternalNodes) { // Internal node?
				// Fetch AABBs of the two child bvh.
				i32 nodeIdx = nodeAddr * 4;
				const ei::Vec4 Lmin_cL = bvh.bvh[nodeIdx];
				const ei::Vec4 Lmax_nL = bvh.bvh[nodeIdx + 1];
				const ei::Vec4 Rmin_cR = bvh.bvh[nodeIdx + 2];
				const ei::Vec4 Rmax_nR = bvh.bvh[nodeIdx + 3];

				// Intersect the ray against the children bounds.
				float c0min, c1min;
				bool traverseChild0 = intersect(ei::Vec3{Lmin_cL}, ei::Vec3{Lmax_nL}, invDir, ood, tmin, tmax, c0min);
				bool traverseChild1 = intersect(ei::Vec3{Rmin_cR}, ei::Vec3{Rmax_nR}, invDir, ood, tmin, tmax, c1min);

				// Neither child was intersected => pop stack.
				if(!traverseChild0 && !traverseChild1) {
					nodeAddr = *stackPtr;
					--stackPtr;
					if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
						primCount = *stackPtr;
						--stackPtr;
					}
				}
				// Otherwise => fetch child pointers.
				else {
					nodeAddr = traverseChild0 ? float_bits_as_int(Lmin_cL.w) : float_bits_as_int(Rmin_cR.w);
					primCount = traverseChild0 ? float_bits_as_int(Lmax_nL.w) : float_bits_as_int(Rmax_nR.w);

					// Both children were intersected => push the farther one.
					if (traverseChild0 && traverseChild1) {
						i32 pushAddr = float_bits_as_int(Rmin_cR.w); // nodeAddr is Lmin_cL.w, this is the other one
						i32 pushCount = float_bits_as_int(Rmax_nR.w);
						if (c1min < c0min) {
							i32 tmp = nodeAddr;
							nodeAddr = pushAddr;
							pushAddr = tmp;
							tmp = primCount;
							primCount = pushCount;
							pushCount = tmp;
						}
						if(pushAddr >= bvh.numInternalNodes) { // Leaf? Then push the count too
							++stackPtr;
							*stackPtr = pushCount;
						}
						++stackPtr;
						*stackPtr = pushAddr;
					}
				}
			}
			
			if(nodeAddr >= bvh.numInternalNodes && nodeAddr != EntrypointSentinel) { // Leaf?
				const i32 instanceId = nodeAddr - bvh.numInternalNodes;

				// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
				for(i32 i = 0; i < primCount; i++) {
					first_intersection_scene_obj_lbvh(scene, ray, startInsPrimId,
						bvh.primIds[ instanceId + i], stackPtr+1,
						hitT, hitInstanceId, hitPrimId, surfParams);
				}

				// Pop next node.
				nodeAddr = *stackPtr;
				--stackPtr;
				if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
					primCount = *stackPtr;
					--stackPtr;
				}
			}
		}
	}

	// Nobody should update hitT if no primitive is hit
	mAssert((hitInstanceId != IGNORE_ID && hitPrimId != IGNORE_ID) || hitT == tmax);

	/* TEST CODE WHICH MAKES A LINEAR TEST (without the BVH)
	for(int i = 0; i < scene.numInstances; ++i) {
		auto& obj = scene.objects[ scene.objectIndices[i] ];
		const ei::Mat3x3 invRotScale = ei::invert(ei::Mat3x3{scene.transformations[i]});
		const ei::Vec3 invTranslation { -scene.transformations[i][3],
										-scene.transformations[i][7],
										-scene.transformations[i][11] };
		ei::Ray transRay = { invRotScale * (ray.origin + invTranslation),
							 normalize(invRotScale * ray.direction) };
		for(int p = 0; p < obj.numPrimitives; ++p) {
			if(intersects_primitve(obj, transRay, p, -1, hitPrimId, hitT, surfParams))
				hitInstanceId = i;
		}
	}*/

	if(hitInstanceId == IGNORE_ID) {
		return { hitT, { IGNORE_ID, IGNORE_ID } };
	} else {
		// To be determined
		ei::Vec3 normal;
		ei::Vec3 tangentX;
		ei::Vec3 tangentY;
		ei::Vec2 uv;

		i32 primId = hitPrimId;


		const ObjectDescriptor<dev>& obj = scene.objects[ scene.objectIndices[hitInstanceId] ];
		const i32 offsetSpheres = obj.polygon.numTriangles + obj.polygon.numQuads;
		if(primId >= offsetSpheres) { // Sphere?
			const i32 sphId = primId - offsetSpheres;
			const ei::Vec3 hitPoint = ray.origin + hitT * ray.direction;
			const Point center { scene.transformations[hitInstanceId] * ei::Vec4(obj.spheres.spheres[sphId].center, 1.0f) };
			normal = normalize(hitPoint - center);

			// Normalization is done later
			if(normal.x == 0.0f && normal.y == 0.0f)
				tangentX = ei::Vec3(1.0f, 0.0f, 0.0f);
			else
				tangentX = normalize(ei::Vec3(ei::Vec2(normal.y, -normal.x), 0.0f));
			tangentY = ei::cross(normal, tangentX);

			uv.x = atan2f(normal.y, normal.x) / (2.0f * ei::PI) + 0.5f;
			uv.y = acosf(normal.z) / ei::PI;
			surfParams.st = uv;
			return RayIntersectionResult{ hitT, { hitInstanceId, hitPrimId }, normal, tangentX, tangentY, uv, surfParams };
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
				float det = 1.f / (du0.x * du1.y - du0.y - du1.x);
				// TODO: fetch the instance instead (issue #44)
				tangentX = det * (dx0 * du1.y - dx1 * du0.y);
				tangentY = det * (dx1 * du0.x - dx0 * du1.x);

				// Don't use the UV tangents to compute the normal, since they may be reversed
				normal = ei::cross(v[1u] - v[0u], v[2u] - v[0u]);
				mAssert(ei::dot(normal, obj.polygon.normals[ids.x]) > 0.f);

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
				// TODO: fetch the instance instead (issue #44)
				const ei::Vec3 dxds = (1.f - surfParams.bilinear.y) * (v[3u] - v[0u]) + surfParams.bilinear.y * (v[2u] - v[1u]);
				const ei::Vec3 dxdt = (1.f - surfParams.bilinear.x) * (v[1u] - v[0u]) + surfParams.bilinear.x * (v[2u] - v[3u]);
				const ei::Matrix<float, 3, 2> dxdst{
					dxds.x, dxdt.x,
					dxds.y, dxdt.y,
					dxds.z, dxdt.z
				};
				const ei::Vec2 duds = (1.f - surfParams.bilinear.y) * (uvV[3u] - uvV[0u]) + surfParams.bilinear.y * (uvV[2u] - uvV[1u]);
				const ei::Vec2 dudt = (1.f - surfParams.bilinear.x) * (uvV[1u] - uvV[0u]) + surfParams.bilinear.x * (uvV[2u] - uvV[3u]);
				const ei::Matrix<float, 2, 2> dudst{
					duds.x, dudt.x,
					duds.y, dudt.y,
				};
				const ei::Mat2x2 dsduv = ei::invert(dudst);
				const ei::Matrix<float, 3, 2> tangents = dxdst * dsduv;
				tangentX = ei::Vec3{ tangents(0, 0), tangents(1, 0), tangents(2, 0) };
				tangentY = ei::Vec3{ tangents(0, 1), tangents(1, 1), tangents(2, 1) };

				// Check if we need to flip the normal (UV coordinates may not coincide with ST)
				if(duds.x*dudt.y - dudt.x*duds.y >= 0)
					normal = ei::cross(tangentY, tangentX);
				else
					normal = ei::cross(tangentX, tangentY);
				mAssert(ei::dot(normal, obj.polygon.normals[ids.x]) > 0.f);
				uv = ei::bilerp(uvV[0u], uvV[1u], uvV[3u], uvV[2u], surfParams.bilinear.x, surfParams.bilinear.y);
			}
		}

		// TODO: enable this for (probably) better code?
		//normal = ei::normalize(ei::transformDir(normal, transforms[hitInstanceId]));
		//tangent = ei::normalize(ei::transformDir(tangent, transforms[hitInstanceId]));
		const ei::Mat3x3 transMatrix = ei::Mat3x3{ scene.transformations[hitInstanceId] };
		normal = ei::normalize(transMatrix * normal);
		tangentX = ei::normalize(transMatrix * tangentX);
		tangentY = ei::normalize(transMatrix * tangentY);

		return RayIntersectionResult{ hitT, { hitInstanceId, hitPrimId }, normal, tangentX, tangentY, uv, surfParams };
	}
}

template < Device dev > CUDA_FUNCTION
bool any_intersection_scene_obj_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const i32 instanceId,
	float tmax,
	i32* traversalStack
) {
	const ei::Mat3x3 invRotScale = ei::invert(ei::Mat3x3{scene.transformations[instanceId]});
	const ei::Vec3 invTranslation { -scene.transformations[instanceId][3],
									-scene.transformations[instanceId][7],
									-scene.transformations[instanceId][11] };
	ei::Ray transRay = { invRotScale * (ray.origin + invTranslation),
						 normalize(invRotScale * ray.direction) };
	const ei::Vec3 invDir = sdiv(1.0f, transRay.direction);
	const ei::Vec3 ood = transRay.origin * invDir;

	const i32 objId = scene.objectIndices[instanceId];
	const ei::Box& box = scene.aabbs[objId];
	const float tmin = 1e-6f * len(box.max - box.min);

	// Intersect the ray against the obj bounding box.
	float hitT;
	if(intersect(box.min, box.max, invDir, ood, tmin, tmax, hitT)) {
		// Intersect the ray against the obj primtive bvh.
		const ObjectDescriptor<dev>& obj = scene.objects[objId];
		const LBVH* lbvh = (LBVH*)obj.accelStruct.accelParameters;
		const i32 checkPrimId = (startInsPrimId.instanceId == instanceId) ? startInsPrimId.primId : IGNORE_ID;
		// Do ray-obj test.
		return any_intersection_obj_lbvh_imp(*lbvh, obj, transRay, checkPrimId,
			invDir, ood, tmin, tmax, traversalStack);
	}
	return false;
}

template < Device dev > CUDA_FUNCTION
bool any_intersection_scene_lbvh_imp(
	const LBVH& bvh,
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
) {
	const ei::Vec3 invDir = sdiv(1.0f, ray.direction);
	const ei::Vec3 ood = ray.origin * invDir;
	const float tmin = 1e-6f * len(scene.aabb.max - scene.aabb.min);

	if(scene.numInstances == 1) {
		i32 traversalStack[OBJ_STACK_SIZE];
		return any_intersection_scene_obj_lbvh(scene, ray, startInsPrimId,
			0, tmax, traversalStack);
	} else {
		// Setup traversal.
		// Traversal stack in CUDA thread-local memory.
		i32 traversalStack[STACK_SIZE];
		traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.
		i32 nodeAddr = 0; // Start from the root.  
		i32* stackPtr = traversalStack; // Current position in traversal stack.
		i32 primCount = 0; // Internal nodes have no primitives

		// Traversal loop.
		while (nodeAddr != EntrypointSentinel) {
			if(nodeAddr < bvh.numInternalNodes) { // Internal node?
				// Fetch AABBs of the two child bvh.
				i32 nodeIdx = nodeAddr * 4;
				const ei::Vec4 Lmin_cL = bvh.bvh[nodeIdx];
				const ei::Vec4 Lmax_nL = bvh.bvh[nodeIdx + 1];
				const ei::Vec4 Rmin_cR = bvh.bvh[nodeIdx + 2];
				const ei::Vec4 Rmax_nR = bvh.bvh[nodeIdx + 3];

				// Intersect the ray against the child bvh.
				float c0min, c1min;
				bool traverseChild0 = intersect(ei::Vec3{Lmin_cL}, ei::Vec3{Lmax_nL}, invDir, ood, tmin, tmax, c0min);
				bool traverseChild1 = intersect(ei::Vec3{Rmin_cR}, ei::Vec3{Rmax_nR}, invDir, ood, tmin, tmax, c1min);

				// Neither child was intersected => pop stack.
				if (!traverseChild0 && !traverseChild1) {
					nodeAddr = *stackPtr;
					--stackPtr;
					if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
						primCount = *stackPtr;
						--stackPtr;
					}
				}
				// Otherwise => fetch child pointers.
				else {
					nodeAddr = traverseChild0 ? float_bits_as_int(Lmin_cL.w) : float_bits_as_int(Rmin_cR.w);
					primCount = traverseChild0 ? float_bits_as_int(Lmax_nL.w) : float_bits_as_int(Rmax_nR.w);

					// Both children were intersected => push the farther one.
					if (traverseChild0 && traverseChild1) {
						i32 pushAddr = float_bits_as_int(Rmin_cR.w); // nodeAddr is Lmin_cL.w, this is the other one
						i32 pushCount = float_bits_as_int(Rmax_nR.w);
						if (c1min < c0min) {
							i32 tmp = nodeAddr;
							nodeAddr = pushAddr;
							pushAddr = tmp;
							tmp = primCount;
							primCount = pushCount;
							pushCount = tmp;
						}
						if(pushAddr >= bvh.numInternalNodes) { // Leaf? Then push the count too
							++stackPtr;
							*stackPtr = pushCount;
						}
						++stackPtr;
						*stackPtr = pushAddr;
					}
				}
			}

			if(nodeAddr >= bvh.numInternalNodes && nodeAddr != EntrypointSentinel) { // Leaf?
				const i32 instanceId = nodeAddr - bvh.numInternalNodes;

				// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
				for(i32 i = 0; i < primCount; i++) {
					if(any_intersection_scene_obj_lbvh(scene, ray, startInsPrimId,
						bvh.primIds[ instanceId + i ], tmax, stackPtr+1))
						return true;
				}

				// Pop next node.
				nodeAddr = *stackPtr;
				--stackPtr;
				if(nodeAddr >= bvh.numInternalNodes) { // Leafs additionally store the primitive count
					primCount = *stackPtr;
					--stackPtr;
				}
			}
		}
		return false;
	}
}

template < Device dev > CUDA_FUNCTION
bool any_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
) {
	const LBVH* lbvh = (const LBVH*)scene.accelStruct.accelParameters;
	return any_intersection_scene_lbvh_imp<dev>(
		*lbvh, scene, ray, startInsPrimId, tmax);
}

template __host__ __device__ bool any_intersection_scene_lbvh(
	const SceneDescriptor<Device::CUDA>& scene,
	const ei::Ray& ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

template __host__ __device__ bool any_intersection_scene_lbvh(
	const SceneDescriptor<Device::CPU>& scene,
	const ei::Ray& ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

template < Device dev > CUDA_FUNCTION
RayIntersectionResult first_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray& ray,
	const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
) {
	const LBVH* lbvh = (const LBVH*)scene.accelStruct.accelParameters;
	return first_intersection_scene_lbvh_imp<dev>(
		*lbvh, scene, ray, startInsPrimId, tmax);
}

template __host__ __device__ RayIntersectionResult first_intersection_scene_lbvh(
	const SceneDescriptor<Device::CUDA>&,
	const ei::Ray&,
	const RayIntersectionResult::HitID&,
	const float
);

template __host__ __device__ RayIntersectionResult first_intersection_scene_lbvh(
	const SceneDescriptor<Device::CPU>& ,
	const ei::Ray&,
	const RayIntersectionResult::HitID&,
	const float
);

}}} // namespace mufflon::scene::accel_struct
