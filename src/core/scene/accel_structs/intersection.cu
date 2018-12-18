#include "intersection.hpp"
#include "accel_structs_commons.hpp"
#include "lbvh.hpp"
#include "util/types.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <ei/3dintersection.hpp>

namespace mufflon {
namespace {

#define STACK_SIZE              96 //64          // Size of the traversal stack in local memory.
#define OBJ_STACK_SIZE              64 //64          // Size of the traversal stack in local memory.
enum : i32
{
	EntrypointSentinel = (i32)0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
	IGNORE_ID = (i32)0xFFFFFFFF,
	SECOND_QUAD_TRIANGLE_BIT = (i32)0x80000000,
	SECOND_QUAD_TRIANGLE_MASK = (i32)0x7FFFFFFF,
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

}
}

namespace mufflon {
namespace scene {
namespace accel_struct {

CUDA_FUNCTION bool interset(const ei::Box box, const ei::Vec3 invDir, const ei::Vec3 ood, 
	const float tmin, const float tmax) {//, float& cmin, float& cmax) {
#ifdef __CUDA_ARCH__
	ei::Vec3 lo = box.min * invDir - ood;
	ei::Vec3 hi = box.max * invDir - ood;
	const float cmin = spanBeginKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
	const float cmax = spanEndKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);
	return cmin <= cmax;
#else
	float t0 = box.min.x * invDir.x - ood.x;
	float t1 = box.max.x * invDir.x - ood.x;
	float cmin = ei::min(t0, t1);
	float cmax = ei::max(t0, t1);
	if (cmax < tmin || cmin > tmax) return false;
	t0 = box.min.y * invDir.y - ood.y;
	t1 = box.max.y * invDir.y - ood.y;
	float min2 = ei::min(t0, t1);
	float max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	if (cmax < tmin || cmin > tmax || cmin > cmax) return false;
	t0 = box.min.z * invDir.z - ood.z;
	t1 = box.max.z * invDir.z - ood.z;
	min2 = ei::min(t0, t1);
	max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	return (cmax >= tmin) && (cmin <= tmax) && (cmin <= cmax);
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION bool interset(const ei::Vec4 xy, const float loz, const float hiz, 
	const ei::Vec3 invDir, const ei::Vec3 ood,
	const float tmin, const float tmax, float& cmin) {
#ifdef __CUDA_ARCH__
	return false;
#else
	float t0 = xy.x * invDir.x - ood.x;
	float t1 = xy.y * invDir.x - ood.x;
	cmin = ei::min(t0, t1);
	float cmax = ei::max(t0, t1);
	if (cmax < tmin || cmin > tmax) return false;
	t0 = xy.z * invDir.y - ood.y;
	t1 = xy.w * invDir.y - ood.y;
	float min2 = ei::min(t0, t1);
	float max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	if (cmax < tmin || cmin > tmax || cmin > cmax) return false;
	t0 = loz * invDir.z - ood.z;
	t1 = hiz * invDir.z - ood.z;
	min2 = ei::min(t0, t1);
	max2 = ei::max(t0, t1);
	cmin = ei::max(cmin, min2);
	cmax = ei::min(cmax, max2);
	return (cmax >= tmin) && (cmin <= tmax) && (cmin <= cmax);
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION void interset_2box(const ei::Vec4 n0xy, const ei::Vec4 n1xy, 
	const ei::Vec4 nz, const ei::Vec3 invDir, const ei::Vec3 ood,
	const float tmin, const float tmax, 
	float& c0min, float& c1min, 
	bool& traverseChild0, bool& traverseChild1) {
#ifdef __CUDA_ARCH__
	// Intersect the ray against the child bvh.
	const float c0lox = n0xy.x * invDir.x - ood.x;
	const float c0hix = n0xy.y * invDir.x - ood.x;
	const float c0loy = n0xy.z * invDir.y - ood.y;
	const float c0hiy = n0xy.w * invDir.y - ood.y;
	const ei::Vec4 c01z = nz * invDir.z - ood.z;
	c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c01z.x, c01z.y, tmin);
	const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c01z.x, c01z.y, tmax);
	const float c1lox = n1xy.x * invDir.x - ood.x;
	const float c1hix = n1xy.y * invDir.x - ood.x;
	const float c1loy = n1xy.z * invDir.y - ood.y;
	const float c1hiy = n1xy.w * invDir.y - ood.y;
	c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c01z.z, c01z.w, tmin);
	const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c01z.z, c01z.w, tmax);
	traverseChild0 = (c0max >= c0min);
	traverseChild1 = (c1max >= c1min);
#else
	traverseChild0 = interset(n0xy, nz.x, nz.y, invDir, ood, tmin, tmax, c0min);
	traverseChild1 = interset(n1xy, nz.y, nz.z, invDir, ood, tmin, tmax, c1min);
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION bool any_intersection_obj_lbvh_imp(
	const ei::Vec4* __restrict__ bvh,
	const i32 bvhSize,
	const ei::Vec3* __restrict__ meshVertices,
	const i32* __restrict__ triIndices,
	const i32* __restrict__ quadIndices,
	const ei::Vec4* __restrict__ spheres,
	const i32 offsetQuads, const i32 offsetSpheres,
	const i32* __restrict__ primIds,
	const i32 numPrimives,
	const ei::Ray ray, const i32 startPrimId,
	const ei::Vec3 invDir,
	const ei::Vec3 ood,
	const float tmin, const float tmax,
	i32* traversalStack
) {
	// Since all threads go to the following branch if numPrimitives == 1,
	// there is no problem with branching.
	if (numPrimives == 1) {
		if (offsetSpheres == 0) {
			const ei::Vec4 v = spheres[0];
			const ei::Sphere sph = { ei::Vec3(v), v.w };
			float t;
			if (ei::intersects(ray, sph, t)) {
				if (t > tmin && t < tmax) {
					return true;
				}
			}
		}
		else if (offsetQuads == 0) {
			int check01 = 3;
			if (startPrimId != IGNORE_ID) {
				if (startPrimId == 0)
					check01 = 0x00000002;
				else if ((startPrimId & SECOND_QUAD_TRIANGLE_MASK) == 0)
					check01 = 0x00000001;
			}
			const ei::Vec3 v[4] = { meshVertices[quadIndices[0]],
						meshVertices[quadIndices[1]],
						meshVertices[quadIndices[2]],
						meshVertices[quadIndices[3]] };
			float t;
			if (check01 & 0x00000001) {
				const ei::Triangle tri0 = { v[0], v[1], v[2] };
				if (ei::intersects(ray, tri0, t)) {
					if (t > tmin && t < tmax) {
						return true;
					}
				}
			}

			if (check01 & 0x00000002) {
				const ei::Triangle tri1 = { v[0], v[2], v[3] };
				if (ei::intersects(ray, tri1, t)) {
					if (t > tmin && t < tmax) {
						return true;
					}
				}
			}
		}
		else {
			const ei::Triangle tri = { meshVertices[triIndices[0]],
							meshVertices[triIndices[0]],
							meshVertices[triIndices[0]] };

			float t;
			if (ei::intersects(ray, tri, t)) {
				if (t > tmin && t < tmax) {
					return true;
				}
			}
		}
		return false;
	}
	// Setup traversal.
	traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.

	// nodeAddr: Non-negative: current internal node; negative: leaf.
	i32 nodeAddr = 0; // Start from the root.  
	char* stackPtr = (char*)&traversalStack[0]; // Current position in traversal stack.

	// Traversal loop.
	while (nodeAddr != EntrypointSentinel)
	{
		// while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
		while (u32(nodeAddr) < u32(EntrypointSentinel))   // functionally equivalent, but faster
		{
			// Fetch AABBs of the two child bvh.
			const ei::Vec4 n0xy = bvh[nodeAddr]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const ei::Vec4 n1xy = bvh[nodeAddr + 1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const ei::Vec4 nz = bvh[nodeAddr + 2];// (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
			ei::Vec4 tmp = bvh[nodeAddr + 3]; // child_index0, child_index1
			int2  cnodes = *(int2*)&tmp;

			// Intersect the ray against the child bvh.
			float c0min, c1min;
			bool traverseChild0, traverseChild1;
			interset_2box(n0xy, n1xy, nz, invDir, ood, tmin, tmax, c0min, c1min, traverseChild0, traverseChild1);

			// Neither child was intersected => pop stack.
			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(i32*)stackPtr;
				stackPtr -= 4;
			}
			// Otherwise => fetch child pointers.
			else {
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

				// Both children were intersected => push the farther one.
				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min) {
						i32 tmp = nodeAddr;
						nodeAddr = cnodes.y;
						cnodes.y = tmp;
					}
					stackPtr += 4;
					*(i32*)stackPtr = cnodes.y;
				}
			}
		}

		// Process postponed leaf bvh.
		// TODO: use warp/block to do the intersection test.
		while (nodeAddr < 0)
		{
			const i32 leafId = ~nodeAddr;
			ei::IVec4 counts; // x: tri; y: quds; z: spheres; w: total count.
			i32 primId;
			i32 startId;
			if (leafId >= bvhSize) {
				startId = leafId - bvhSize;
				primId = primIds[startId];
				if (primId >= offsetSpheres) {
					counts = ei::IVec4(0, 0, 1, 1);
				}
				else if (primId >= offsetQuads) {
					counts = ei::IVec4(0, 1, 0, 1);
				}
				else
					counts = ei::IVec4(1, 0, 0, 1);
			}
			else {
				const ei::IVec4 leaf = ((ei::IVec4*)bvh)[leafId];
				// Extract counts for three kinds of primitvies.
				extract_prim_counts(leaf.x, counts);
				startId = leaf.y;
				primId = primIds[startId++];
			}
			// Triangles intersetion test.
			// Uses Moeller-Trumbore intersection algorithm:
			// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
			for (i32 i = 0; i < counts.x; i++)
			{
				if (primId == startPrimId)
					continue;
				const i32 triId = primId * 3;
				const ei::Triangle tri = { meshVertices[triIndices[triId]],
							meshVertices[triIndices[triId + 1]],
							meshVertices[triIndices[triId + 2]] };

				float t;
				if (ei::intersects(ray, tri, t)) {
					if (t > tmin && t < tmax) {
						return true;
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}


			// Quads intersetion test.
			for (i32 i = 0; i < counts.y; i++)
			{
				int check01 = 3;
				if (startPrimId != IGNORE_ID) {
					if (startPrimId == primId) 
						check01 = 0x00000002;
					else if ((startPrimId & SECOND_QUAD_TRIANGLE_MASK) == primId)
						check01 = 0x00000001;
				}
				const i32 quadId = (primId - offsetQuads) * 4;
				const ei::Vec3 v[4] = { meshVertices[quadIndices[quadId]],
							meshVertices[quadIndices[quadId + 1]],
							meshVertices[quadIndices[quadId + 2]],
							meshVertices[quadIndices[quadId + 3]] };
				float t;
				if (check01 & 0x00000001) {
					const ei::Triangle tri0 = { v[0], v[1], v[2] };
					if (ei::intersects(ray, tri0, t)) {
						if (t > tmin && t < tmax) {
							return true;
						}
					}
				}

				if (check01 & 0x00000002) {
					const ei::Triangle tri1 = { v[0], v[2], v[3] };
					if (ei::intersects(ray, tri1, t)) {
						if (t > tmin && t < tmax) {
							return true;
						}
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}


			// Spheres intersetion test.
			for (i32 i = 0; i < counts.z; i++)
			{
				const i32 sphId = primId - offsetSpheres;
				const ei::Vec4 v = spheres[sphId];
				const ei::Sphere sph = { ei::Vec3(v), v.w };
				float t;
				if (ei::intersects(ray, sph, t)) {
					if (t > tmin && t < tmax) {
						return true;
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}
			// Postponed next node.
			nodeAddr = *(i32*)stackPtr;
			stackPtr -= 4;
		}
	}
	return false;
}


CUDA_FUNCTION void first_intersection_obj_lbvh_imp(
	const ei::Vec4* __restrict__ bvh,
	const i32 bvhSize,
	const ei::Vec3* __restrict__ meshVertices,
	const ei::Vec2* __restrict__ meshUVs,
	const i32* __restrict__ triIndices,
	const i32* __restrict__ quadIndices,
	const ei::Vec4* __restrict__ spheres,
	const i32 offsetQuads, const i32 offsetSpheres,
	const i32* __restrict__ primIds,
	const i32 numPrimives,
	const ei::Ray ray, const i32 startPrimId,
	const ei::Vec3 invDir, 
	const ei::Vec3 ood,
	const float tmin, const float tmax,
	int& hitPrimId, float& hitT,
	ei::Vec3& hitBarycentric,
	i32* traversalStack
) {
	if (numPrimives == 1) {
		if (offsetSpheres == 0) {
			const ei::Vec4 v = spheres[0];
			const ei::Sphere sph = { ei::Vec3(v), v.w };
			float t;
			if (ei::intersects(ray, sph, t)) {
				if (t > tmin && t < hitT) {
					hitT = t;
					hitPrimId = 0;
				}
			}
		}
		else if (offsetQuads == 0) {
			int check01 = 3;
			if (startPrimId != IGNORE_ID) {
				if (startPrimId == 0)
					check01 = 0x00000002;
				else if ((startPrimId & SECOND_QUAD_TRIANGLE_MASK) == 0)
					check01 = 0x00000001;
			}
			const ei::Vec3 v[4] = { meshVertices[quadIndices[0]],
						meshVertices[quadIndices[1]],
						meshVertices[quadIndices[2]],
						meshVertices[quadIndices[3]] };
			float t;
			ei::Vec3 barycentric;
			if (check01 & 0x00000001) {
				const ei::Triangle tri0 = { v[0], v[1], v[2] };
				if (ei::intersects(ray, tri0, t, barycentric)) {
					if (t > tmin && t < hitT) {
						hitT = t;
						hitPrimId = 0;
						hitBarycentric = barycentric;
					}
				}
			}

			if (check01 & 0x00000002) {
				const ei::Triangle tri1 = { v[0], v[2], v[3] };
				if (ei::intersects(ray, tri1, t, barycentric)) {
					if (t > tmin && t < hitT) {
						hitT = t;
						hitPrimId = SECOND_QUAD_TRIANGLE_BIT;
						hitBarycentric = barycentric;
					}
				}
			}
		}
		else {
			const ei::Triangle tri = { meshVertices[triIndices[0]],
							meshVertices[triIndices[1]],
							meshVertices[triIndices[2]] };

			float t;
			ei::Vec3 barycentric;
			if (ei::intersects(ray, tri, t, barycentric)) {
				if (t > tmin && t < hitT) {
					hitT = t;
					hitPrimId = 0;
					hitBarycentric = barycentric;
				}
			}
		}
		return;
	} 
	// Setup traversal.
	traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.

	// nodeAddr: Non-negative: current internal node; negative: leaf.
	i32 nodeAddr = 0; // Start from the root.  
	char* stackPtr = (char*)&traversalStack[0]; // Current position in traversal stack.

	// Traversal loop.
	while (nodeAddr != EntrypointSentinel)
	{
		// while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
		while (u32(nodeAddr) < u32(EntrypointSentinel))   // functionally equivalent, but faster
		{
			// Fetch AABBs of the two child bvh.
			const ei::Vec4 n0xy = bvh[nodeAddr]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const ei::Vec4 n1xy = bvh[nodeAddr + 1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const ei::Vec4 nz = bvh[nodeAddr + 2];// (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
			ei::Vec4 tmp = bvh[nodeAddr + 3]; // child_index0, child_index1
			int2  cnodes = *(int2*)&tmp;

			// Intersect the ray against the child bvh.
			float c0min, c1min;
			bool traverseChild0, traverseChild1;
			interset_2box(n0xy, n1xy, nz, invDir, ood, tmin, hitT, c0min, c1min, traverseChild0, traverseChild1);

			// Neither child was intersected => pop stack.
			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(i32*)stackPtr;
				stackPtr -= 4;
			}
			// Otherwise => fetch child pointers.
			else {
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

				// Both children were intersected => push the farther one.
				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min) {
						i32 tmp = nodeAddr;
						nodeAddr = cnodes.y;
						cnodes.y = tmp;
					}
					stackPtr += 4;
					*(i32*)stackPtr = cnodes.y;
				}
			}
		}

		// Process postponed leaf bvh.
		// TODO: use warp/block to do the intersection test.
		while (nodeAddr < 0)
		{
			const i32 leafId = ~nodeAddr;
			ei::IVec4 counts; // x: tri; y: quds; z: spheres; w: total count.
			i32 primId;
			i32 startId;
			if (leafId >= bvhSize) {
				startId = leafId - bvhSize;
				primId = primIds[startId];
				if (primId >= offsetSpheres) {
					counts = ei::IVec4(0, 0, 1, 1);
				}
				else if (primId >= offsetQuads) {
					counts = ei::IVec4(0, 1, 0, 1);
				}
				else
					counts = ei::IVec4(1, 0, 0, 1);
			}
			else {
				const ei::IVec4 leaf = ((ei::IVec4*)bvh)[leafId];
				// Extract counts for three kinds of primitvies.
				extract_prim_counts(leaf.x, counts);
				startId = leaf.y;
				primId = primIds[startId++];
			}
			// Triangles intersetion test.
			// Uses Moeller-Trumbore intersection algorithm:
			// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
			for (i32 i = 0; i < counts.x; i++)
			{
				if (primId == startPrimId)
					continue;
				const i32 triId = primId * 3;
				const ei::Triangle tri = { meshVertices[triIndices[triId]],
							meshVertices[triIndices[triId + 1]],
							meshVertices[triIndices[triId + 2]] };

				float t;
				ei::Vec3 barycentric;
				if (ei::intersects(ray, tri, t, barycentric)) {
					if (t > tmin && t < hitT) {
						hitT = t;
						hitPrimId = primId;
						hitBarycentric = barycentric;
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}


			// Quads intersetion test.
			for (i32 i = 0; i < counts.y; i++)
			{
				int check01 = 3;
				if (startPrimId != IGNORE_ID) {
					if (startPrimId == primId)
						check01 = 0x00000002;
					else if ((startPrimId & SECOND_QUAD_TRIANGLE_MASK) == primId)
						check01 = 0x00000001;
				}
				const i32 quadId = (primId - offsetQuads) * 4;
				const ei::Vec3 v[4] = { meshVertices[quadIndices[quadId]],
							meshVertices[quadIndices[quadId + 1]],
							meshVertices[quadIndices[quadId + 2]],
							meshVertices[quadIndices[quadId + 3]] };
				float t;
				ei::Vec3 barycentric;
				if (check01 & 0x00000001) {
					const ei::Triangle tri0 = { v[0], v[1], v[2] };
					if (ei::intersects(ray, tri0, t, barycentric)) {
						if (t > tmin && t < hitT) {
							hitT = t;
							hitPrimId = primId;
							hitBarycentric = barycentric;
						}
					}
				}

				if (check01 & 0x00000002) {
					const ei::Triangle tri1 = { v[0], v[2], v[3] };
					if (ei::intersects(ray, tri1, t, barycentric)) {
						if (t > tmin && t < hitT) {
							hitT = t;
							hitPrimId = primId | SECOND_QUAD_TRIANGLE_BIT;
							hitBarycentric = barycentric;
						}
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}


			// Spheres intersetion test.
			for (i32 i = 0; i < counts.z; i++)
			{
				const i32 sphId = primId - offsetSpheres;
				const ei::Vec4 v = spheres[sphId];
				const ei::Sphere sph = { ei::Vec3(v), v.w };
				float t;
				if (ei::intersects(ray, sph, t)) {
					if (t > tmin && t < hitT) {
						hitT = t;
						hitPrimId = primId;
					}
				}
				if (startId < numPrimives)
					primId = primIds[startId++];
			}
			// Postponed next node.
			nodeAddr = *(i32*)stackPtr;
			stackPtr -= 4;
		}
	}	
}


// TODO: otpimize this.
CUDA_FUNCTION ei::Mat4x4 expand_mat3x4(const ei::Mat3x4 v) {
	//return ei::Mat4x4{ v(0),v(1),v(2),ei::Vec4{0., 0., 0., 1.f } };
	return ei::Mat4x4{ v[0],v[1],v[2],v[3],
		v[4],v[5],v[6],v[7],
		v[8],v[9],v[10],v[11],
		0., 0., 0., 1.f };
}

template < Device dev >
CUDA_FUNCTION
void first_intersection_scene_obj_lbvh(
	const ei::Mat3x4* __restrict__ transforms,
	const i32* __restrict__ objIds,
	const ei::Box* __restrict__ aabbs,
	const ObjectDescriptor<dev>* __restrict__ objs,
	const ei::Ray ray, 
	const float tmin,
	const i32 startInstanceId,
	const i32 startPrimId,
	const i32 instanceId,
	i32* traversalStack,
	float& hitT,
	i32& hitInstanceId,
	i32& hitPrimId,
	ei::Vec3& hitBarycentric
) {
	const ei::Mat4x4 transMatrix = expand_mat3x4(transforms[instanceId]);
	const ei::Mat4x4 invMatrix = ei::invert(transMatrix);
	ei::Ray transRay = { ei::Vec3{invMatrix * ei::Vec4{ray.origin, 1.f}},
		ei::Mat3x3{invMatrix} *ray.direction };
	float invScale = 1.f / ei::len(transRay.direction);
	transRay.direction = transRay.direction * invScale;
	const ei::Vec3 invDir = sdiv(1.0f, transRay.direction);
	const ei::Vec3 ood = transRay.origin * invDir;

	const i32 objId = objIds[instanceId];
	const ei::Box box = aabbs[objId];

	// Intersect the ray against the obj bounding box.
	if (interset(box, invDir, ood, tmin, hitT)) {
		// Intersect the ray against the obj primtive bvh.
		const ObjectDescriptor<dev>& obj = objs[objId];
		LBVH* lbvh = (LBVH*)obj.accelStruct.accelParameters;
		const i32 numFaces = obj.polygon.numTriangles + obj.polygon.numQuads;
		const i32 checkPrimId = (startInstanceId == instanceId) ? startPrimId : IGNORE_ID;
		i32 primId = IGNORE_ID;
		float t = hitT;
		ei::Vec3 barycentric;
		const i32 offsetSphere = numFaces + obj.spheres.numSpheres;
		// Do ray-obj test.
		first_intersection_obj_lbvh_imp(
			lbvh->bvh,
			lbvh->bvhSize,
			obj.polygon.vertices,
			obj.polygon.uvs,
			(i32*)obj.polygon.vertexIndices,
			(i32*)(obj.polygon.vertexIndices + obj.polygon.numTriangles),
			(ei::Vec4*)obj.spheres.spheres,
			obj.polygon.numTriangles,
			numFaces,
			lbvh->primIds,
			offsetSphere,
			transRay,
			checkPrimId,
			invDir,
			ood,
			tmin, hitT,
			primId, t,
			barycentric,
			(i32*)(traversalStack + 4)
		);

		if (primId != IGNORE_ID) {
			// Set transformed t.
			hitPrimId = primId;
			hitInstanceId = instanceId;
			hitT = t * invScale;
			//if (primId < offsetSphere) // TODO enable this?
			hitBarycentric = barycentric;
		}
	}
}

template < Device dev >
CUDA_FUNCTION
void first_intersection_scene_lbvh_imp(
	const ei::Vec4* __restrict__ bvh,
	const i32 bvhSize,
	const ei::Mat3x4* __restrict__ transforms,
	const i32* __restrict__ objIds,
	const ei::Box* __restrict__ aabbs,
	const i32* __restrict__ instanceIds,
	const i32 numInstances,
	const ObjectDescriptor<dev>* __restrict__ objs,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	float tmaxValue,
	RayIntersectionResult& result
) {
	// Primitive index of the closest intersection, -1 if none.
	const i32 startInstanceId = startInsPrimId.instanceId;
	const i32 startPrimId = startInsPrimId.primId;
	const float tmin = 0.f;
	// TODO adjust tmax with a small value based on scene size.
	const float	tmax = tmaxValue + exp2f(-80.0f);
	i32 hitPrimId = SECOND_QUAD_TRIANGLE_BIT;						// No primitive intersected so far.
	i32 hitInstanceId = -1;
	ei::Vec3 hitBarycentric;
	float hitT = tmax;						// t-value of the closest intersection.
	const ei::Vec3 invDir = sdiv(1.0f, ray.direction);
	const ei::Vec3 ood = ray.origin * invDir;

	if (numInstances == 1) {
		i32 traversalStack[OBJ_STACK_SIZE];
		first_intersection_scene_obj_lbvh(transforms, objIds, aabbs, objs, ray, tmin,
			startInstanceId, startPrimId, 0, traversalStack, hitT, hitInstanceId, hitPrimId,
			hitBarycentric);
	}
	else {
		// Setup traversal.
		// Traversal stack in CUDA thread-local memory.
		i32 traversalStack[STACK_SIZE];
		traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.

		// nodeAddr: Non-negative: current internal node; negative: leaf.
		i32 nodeAddr = 0; // Start from the root.  
		char* stackPtr = (char*)&traversalStack[0]; // Current position in traversal stack.

		// Traversal loop.
		while (nodeAddr != EntrypointSentinel)
		{
			// while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
			while (u32(nodeAddr) < u32(EntrypointSentinel))   // functionally equivalent, but faster
			{
				// Fetch AABBs of the two child bvh.
				const ei::Vec4 n0xy = bvh[nodeAddr]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const ei::Vec4 n1xy = bvh[nodeAddr + 1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const ei::Vec4 nz = bvh[nodeAddr + 2];// (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				ei::Vec4 tmp = bvh[nodeAddr + 3]; // child_index0, child_index1
				int2  cnodes = *(int2*)&tmp;

				// Intersect the ray against the child bvh.
				float c0min, c1min;
				bool traverseChild0, traverseChild1;
				interset_2box(n0xy, n1xy, nz, invDir, ood, tmin, hitT, c0min, c1min, traverseChild0, traverseChild1);

				// Neither child was intersected => pop stack.
				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *(i32*)stackPtr;
					stackPtr -= 4;
				}
				// Otherwise => fetch child pointers.
				else {
					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.
					if (traverseChild0 && traverseChild1)
					{
						if (c1min < c0min) {
							i32 tmp = nodeAddr;
							nodeAddr = cnodes.y;
							cnodes.y = tmp;
						}
						stackPtr += 4;
						*(i32*)stackPtr = cnodes.y;
					}
				}
			}

			// Process postponed leaf bvh.
			// TODO: use warp/block to do the intersection test.
			while (nodeAddr < 0)
			{
				const i32 leafId = ~nodeAddr;
				i32 numCheckInstances;
				i32 instanceId;
				i32 startId;
				if (leafId >= bvhSize) {
					startId = leafId - bvhSize;
					instanceId = instanceIds[startId];
					numCheckInstances = 1;
				}
				else {
					const ei::IVec4 leaf = ((ei::IVec4*)bvh)[leafId];
					numCheckInstances = leaf.x;
					startId = leaf.y;
					instanceId = instanceIds[startId++];
				}

				for (i32 i = 0; i < numCheckInstances; i++)
				{
					first_intersection_scene_obj_lbvh(transforms, objIds, aabbs, objs, ray, tmin,
						startInstanceId, startPrimId, instanceId, (i32*)(stackPtr + 4), hitT, hitInstanceId, hitPrimId,
						hitBarycentric);

					if (startId < numInstances)
						instanceId = instanceIds[startId++];
				}

				// Postponed next node.
				nodeAddr = *(i32*)stackPtr;
				stackPtr -= 4;
			}
		}
	}

	if (hitInstanceId == IGNORE_ID) {
		result = { hitT, { IGNORE_ID, IGNORE_ID } };
	}
	else {
		const i32 objId = objIds[hitInstanceId];
		ObjectDescriptor<dev> obj = objs[objId];
		const i32 offsetQuads = obj.polygon.numTriangles;
		const i32 offsetSpheres = offsetQuads + obj.polygon.numQuads;
		const ei::Vec3* meshVertices = obj.polygon.vertices;
		const ei::Vec2* meshUVs = obj.polygon.uvs;

		ei::Vec3 normal;
		ei::Vec3 tangent;
		ei::Vec2 uv;
		i32 hitSecondTri = 0;
		i32 primId = hitPrimId;
		if (hitPrimId & SECOND_QUAD_TRIANGLE_BIT) {
			primId = hitPrimId & SECOND_QUAD_TRIANGLE_MASK;
			hitSecondTri = 1;
		}

		if (primId < offsetSpheres) {
			const i32* indices = (i32*)obj.polygon.vertexIndices;
			i32 triId;
			if (primId < offsetQuads) {
				// Triangle.
				triId = primId * 3;
			}
			else {
				// Quad.
				triId = (primId - offsetQuads) * 4;
				indices += offsetQuads;
			}
			ei::IVec3 ids = { indices[triId],
			hitSecondTri? indices[triId + 3] : indices[triId + 1],
			indices[triId + 2] };

			ei::Vec3 v[3] = { meshVertices[ids.x], meshVertices[ids.y], meshVertices[ids.z] };
			tangent = v[1] - v[0];
			normal = ei::cross(v[0] - v[2], tangent);

			ei::Vec2 uvV[3] = { meshUVs[ids.x], meshUVs[ids.y], meshUVs[ids.z] };
			uv = uvV[0] * hitBarycentric.x + uvV[1] * hitBarycentric.y +
				uvV[2] * hitBarycentric.z;
		}
		else {
			// Sphere.
			const i32 sphId = primId - offsetSpheres;
			const ei::Vec3 hitPoint = ray.origin + hitT * ray.direction;
			normal = ei::normalize(hitPoint - obj.spheres.spheres[sphId].center);

			if (normal.x == 0.f && normal.y == 0.f) {
				tangent = ei::Vec3(1.f, 0.f, 0.f);
			}
			else {
				// TODO: does normal need to be normalized if we normalize later?
				tangent = ei::Vec3(ei::Vec2(normal.y, -normal.x), 0.f);
			}

			uv.x = atan2f(normal.x, normal.y) / (2.f * ei::PI) + 0.5f;
			uv.y = 0.5f * normal.z + 0.5f;
		}

		// TODO: enable this for (probably) better code?
		//normal = ei::normalize(ei::transformDir(normal, transforms[hitInstanceId]));
		//tangent = ei::normalize(ei::transformDir(tangent, transforms[hitInstanceId]));
		const ei::Mat3x3 transMatrix = ei::Mat3x3{ transforms[hitInstanceId] };
		normal = ei::normalize(transMatrix * normal);
		tangent = ei::normalize(transMatrix * tangent);

		result = { hitT, { hitInstanceId, hitPrimId }, normal, tangent, uv, hitBarycentric };
	}
}

template < Device dev >
CUDA_FUNCTION
bool any_intersection_scene_obj_lbvh(
	const ei::Mat3x4* __restrict__ transforms,
	const i32* __restrict__ objIds,
	const ei::Box* __restrict__ aabbs,
	const ObjectDescriptor<dev>* __restrict__ objs,
	const ei::Ray ray,
	const float tmin,
	const float tmax,
	const i32 startInstanceId,
	const i32 startPrimId,
	const i32 instanceId,
	i32* traversalStack
) {
	const ei::Mat4x4 transMatrix = expand_mat3x4(transforms[instanceId]);
	const ei::Mat4x4 invMatrix = ei::invert(transMatrix);
	ei::Ray transRay = { ei::Vec3{invMatrix * ei::Vec4{ray.origin, 1.f}},
		ei::Mat3x3{invMatrix} *ray.direction };
	float invScale = 1.f / ei::len(transRay.direction);
	transRay.direction = transRay.direction * invScale;
	const ei::Vec3 invDir = sdiv(1.0f, transRay.direction);
	const ei::Vec3 ood = transRay.origin * invDir;

	const i32 objId = objIds[instanceId];
	const ei::Box box = aabbs[objId];

	// Intersect the ray against the obj bounding box.
	if (interset(box, invDir, ood, tmin, tmax)) {
		// Intersect the ray against the obj primtive bvh.
		ObjectDescriptor<dev> obj = objs[objId];
		LBVH* lbvh = (LBVH*)obj.accelStruct.accelParameters;
		const i32 numFaces = obj.polygon.numTriangles + obj.polygon.numQuads;
		const i32 checkPrimId = (startInstanceId == instanceId) ? startPrimId : IGNORE_ID;
		// Do ray-obj test.
		if (any_intersection_obj_lbvh_imp(
			lbvh->bvh,
			lbvh->bvhSize,
			obj.polygon.vertices,
			(i32*)obj.polygon.vertexIndices,
			(i32*)(obj.polygon.vertexIndices + obj.polygon.numTriangles),
			(ei::Vec4*)obj.spheres.spheres,
			obj.polygon.numTriangles,
			numFaces,
			lbvh->primIds,
			numFaces + obj.spheres.numSpheres,
			transRay,
			checkPrimId,
			invDir,
			ood,
			tmin, tmax,
			traversalStack
		))
			return true;
	}
	return false;
}

template < Device dev >
CUDA_FUNCTION
bool any_intersection_scene_lbvh_imp(
	const ei::Vec4* __restrict__ bvh,
	const i32 bvhSize,
	const ei::Mat3x4* __restrict__ transforms,
	const i32* __restrict__ objIds,
	const ei::Box* __restrict__ aabbs,
	const i32* __restrict__ instanceIds,
	const i32 numInstances,
	const ObjectDescriptor<dev>* __restrict__ objs,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmaxValue
) {
	const float tmin = 0.f;
	// TODO adjust tmax with a small value based on scene size.
	const float	tmax = tmaxValue + exp2f(-80.0f); 
	// Primitive index of the closest intersection, -1 if none.
	const i32 startInstanceId = startInsPrimId.instanceId;
	const i32 startPrimId = startInsPrimId.primId;

	const ei::Vec3 invDir = sdiv(1.0f, ray.direction);
	const ei::Vec3 ood = ray.origin * invDir;

	if (numInstances == 1) {
		i32 traversalStack[OBJ_STACK_SIZE];
		return any_intersection_scene_obj_lbvh(transforms, objIds, aabbs, objs, ray,
				tmin, tmax, startInstanceId, startPrimId, 0, traversalStack);
	}
	else {
		// Setup traversal.
		// Traversal stack in CUDA thread-local memory.
		i32 traversalStack[STACK_SIZE];
		traversalStack[0] = EntrypointSentinel;	// Bottom-most entry.

		// nodeAddr: Non-negative: current internal node; negative: leaf.
		i32 nodeAddr = 0; // Start from the root.  
		char* stackPtr = (char*)&traversalStack[0]; // Current position in traversal stack.

		// Traversal loop.
		while (nodeAddr != EntrypointSentinel)
		{
			// while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
			while (u32(nodeAddr) < u32(EntrypointSentinel))   // functionally equivalent, but faster
			{
				// Fetch AABBs of the two child bvh.
				const ei::Vec4 n0xy = bvh[nodeAddr]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const ei::Vec4 n1xy = bvh[nodeAddr + 1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const ei::Vec4 nz = bvh[nodeAddr + 2];// (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				ei::Vec4 tmp = bvh[nodeAddr + 3]; // child_index0, child_index1
				int2  cnodes = *(int2*)&tmp;

				// Intersect the ray against the child bvh.
				float c0min, c1min;
				bool traverseChild0, traverseChild1;
				interset_2box(n0xy, n1xy, nz, invDir, ood, tmin, tmax, c0min, c1min, traverseChild0, traverseChild1);

				// Neither child was intersected => pop stack.
				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *(i32*)stackPtr;
					stackPtr -= 4;
				}
				// Otherwise => fetch child pointers.
				else {
					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.
					if (traverseChild0 && traverseChild1)
					{
						if (c1min < c0min) {
							i32 tmp = nodeAddr;
							nodeAddr = cnodes.y;
							cnodes.y = tmp;
						}
						stackPtr += 4;
						*(i32*)stackPtr = cnodes.y;
					}
				}
			}

			// Process postponed leaf bvh.
			// TODO: use warp/block to do the intersection test.
			while (nodeAddr < 0)
			{
				const i32 leafId = ~nodeAddr;
				i32 numCheckInstances; // 
				i32 instanceId;
				i32 startId;
				if (leafId >= bvhSize) {
					startId = leafId - bvhSize;
					instanceId = instanceIds[startId];
					numCheckInstances = 1;
				}
				else {
					const ei::IVec4 leaf = ((ei::IVec4*)bvh)[leafId];
					numCheckInstances = leaf.x;
					startId = leaf.y;
					instanceId = instanceIds[startId++];
				}

				for (i32 i = 0; i < numCheckInstances; i++)
				{
					if (any_intersection_scene_obj_lbvh(transforms, objIds, aabbs, objs, ray,
						tmin, tmax, startInstanceId, startPrimId, instanceId, (i32*)(stackPtr + 4)))
						return true;

					if (startId < numInstances)
						instanceId = instanceIds[startId++];
				}

				// Postponed next node.
				nodeAddr = *(i32*)stackPtr;
				stackPtr -= 4;
			}
		}
		return false;
	}
}

template < Device dev >
CUDA_FUNCTION
bool any_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
) {
	const LBVH* lbvh = (const LBVH*)scene.accelStruct.accelParameters;
	return any_intersection_scene_lbvh_imp<dev>(		
		(const ei::Vec4*)lbvh->bvh,
		lbvh->bvhSize,
		(const ei::Mat3x4*)scene.transformations,
		(const i32*)scene.objectIndices,
		(const ei::Box*)scene.aabbs,
		(const i32*)lbvh->primIds,
		scene.numInstances,
		scene.objects,
		ray, startInsPrimId, tmax
		);
}

template __host__ __device__ bool any_intersection_scene_lbvh(
	const SceneDescriptor<Device::CUDA>& scene,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

template __host__ __device__ bool any_intersection_scene_lbvh(
	const SceneDescriptor<Device::CPU>& scene,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax
);

template < Device dev >
CUDA_FUNCTION
void first_intersection_scene_lbvh(
	const SceneDescriptor<dev>& scene,
	const ei::Ray ray, const RayIntersectionResult::HitID& startInsPrimId,
	const float tmax, RayIntersectionResult& result
) {
	const LBVH* lbvh = (const LBVH*)scene.accelStruct.accelParameters;
	first_intersection_scene_lbvh_imp<dev>(
		(const ei::Vec4*)lbvh->bvh,
		lbvh->bvhSize,
		(const ei::Mat3x4*)scene.transformations,
		(const i32*)scene.objectIndices,
		(const ei::Box*)scene.aabbs,
		(const i32*)lbvh->primIds,
		scene.numInstances,
		scene.objects,
		ray, startInsPrimId, tmax, result
		);
}

template __host__ __device__ void first_intersection_scene_lbvh(
	const SceneDescriptor<Device::CUDA>&,
	const ei::Ray, const RayIntersectionResult::HitID&,
	const float, RayIntersectionResult&
);

template __host__ __device__ void first_intersection_scene_lbvh(
	const SceneDescriptor<Device::CPU>& ,
	const ei::Ray , const RayIntersectionResult::HitID&,
	const float , RayIntersectionResult&
);

}
}
}