#include "intersection.hpp"
#include "accel_structs_commons.hpp"
#include "util/types.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <ei/3dintersection.hpp>

namespace mufflon {
namespace {

#define STACK_SIZE              64          // Size of the traversal stack in local memory.
enum
{
	EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
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

__global__
void intersection_testD(const ei::Ray ray, const i32 startPrimId, 
	const ei::Vec4* __restrict__ bvh,
	const ei::Vec3* __restrict__ triVertices,
	const ei::Vec3* __restrict__ quadVertices,
	const ei::Vec4* __restrict__ sphVertices,
	const i32* __restrict__ triIndices,
	const i32* __restrict__ quadIndices,
	const i32* __restrict__ primIds,
	const i32 offsetQuads, const i32 offsetSpheres,
	RayIntersectionResult* __restrict__ result,
	const float tmin, const float tmax) {
	// TODO: two problems need to be solved:
	// 1. Severity	Code	Description	Project	File	Line	Suppression State
	// Error		identifier "ei::EPSILON" is undefined in device code	core	E : \works\mufflonTemp\temp1\mufflon\deps\epsilon\include\ei\3dintersection.hpp	699
	// 2. Severity	Code	Description	Project	File	Line	Suppression State
	// Warning		calling a __host__ function from a __host__ __device__ function is not allowed	core	E : \works\mufflonTemp\temp1\mufflon\deps\epsilon\include\ei\3dintersection.hpp	502

#if 0

	// Setup traversal.
	// Traversal stack in CUDA thread-local memory.
	int traversalStack[STACK_SIZE];
	traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

	int     hitPrimId;                       // Primitive index of the closest intersection, -1 if none.
	float   hitT;                           // t-value of the closest intersection.

	// nodeAddr: Non-negative: current internal node; negative: leaf.
	int     nodeAddr = 0; // Start from the root.  
	char*   stackPtr = (char*)&traversalStack[0]; //Current position in traversal stack.
	const float	ooeps = exp2f(-80.0f); // Avoid div by zero.

	const float idirx = 1.0f / (fabsf(ray.direction.x) > ooeps ? ray.direction.x : copysignf(ooeps, ray.direction.x));
	const float idiry = 1.0f / (fabsf(ray.direction.y) > ooeps ? ray.direction.y : copysignf(ooeps, ray.direction.y));
	const float idirz = 1.0f / (fabsf(ray.direction.z) > ooeps ? ray.direction.z : copysignf(ooeps, ray.direction.z));
	const float oodx = ray.origin.x * idirx;
	const float oody = ray.origin.y * idiry;
	const float oodz = ray.origin.z * idirz;

	hitT = tmax;
	nodeAddr = 0;   // Start from the root.
	hitPrimId = -1;  // No primitive intersected so far.

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
			const float c0lox = n0xy.x * idirx - oodx;
			const float c0hix = n0xy.y * idirx - oodx;
			const float c0loy = n0xy.z * idiry - oody;
			const float c0hiy = n0xy.w * idiry - oody;
			const float c0loz = nz.x   * idirz - oodz;
			const float c0hiz = nz.y   * idirz - oodz;
			const float c1loz = nz.z   * idirz - oodz;
			const float c1hiz = nz.w   * idirz - oodz;
			const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
			const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
			const float c1lox = n1xy.x * idirx - oodx;
			const float c1hix = n1xy.y * idirx - oodx;
			const float c1loy = n1xy.z * idiry - oody;
			const float c1hiy = n1xy.w * idiry - oody;
			const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

			const bool traverseChild0 = (c0max >= c0min);
			const bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.
			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
			// Otherwise => fetch child pointers.
			else {
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

				// Both children were intersected => push the farther one.
				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min) {
						int tmp = nodeAddr;
						nodeAddr = cnodes.y;
						cnodes.y = tmp;
					}
					stackPtr += 4;
					*(int*)stackPtr = cnodes.y;
				}
			}
		}

		// Process postponed leaf bvh.
		// TODO: use warp/block to do the intersection test.
		while (nodeAddr < 0)
		{
			const int leafId = ~nodeAddr;
			{
				const ei::Vec4 leaf = bvh[leafId];
				ei::IVec4 counts; // x: tri; y: quds; z: spheres; w: total count.
				// Extract counts for three kinds of primitvies.
				extract_prim_counts(leaf.x, counts);

				i32 startId = leaf.y;
				// Triangles intersetion test.
				// Uses Moeller-Trumbore intersection algorithm:
				// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
				for (i32 i = 0; i < counts.x; i++)
				{
					const i32 primId = primIds[startId++];
					if (primId == startPrimId)
						continue;
					const i32 triId = primId * 3;
					const ei::Triangle tri = { triVertices[triIndices[triId]],
							  triVertices[triIndices[triId + 1]],
							  triVertices[triIndices[triId + 2]] };

					float t;
					if (ei::intersects(ray, tri, t)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
						}
					}
				}


				// Quads intersetion test.
				for (i32 i = 0; i < counts.y; i++)
				{
					const i32 primId = primIds[startId++];
					if (primId == startPrimId)
						continue;
					const i32 quadId = (primId - offsetQuads) * 4;
					const ei::Vec3 v[4] = { quadVertices[quadIndices[quadId]],
							  triVertices[quadIndices[quadId + 1]],
							  triVertices[quadIndices[quadId + 2]],
							  quadVertices[quadIndices[quadId]] };
					const ei::Triangle tri0 = { v[0], v[1], v[2] };
					float t;
					if (ei::intersects(ray, tri0, t)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
						}
					}

					const ei::Triangle tri1 = { v[0], v[1], v[2] };
					if (ei::intersects(ray, tri1, t)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
						}
					}
				}


				// Spheres intersetion test.
				for (i32 i = 0; i < counts.z; i++)
				{
					const i32 primId = primIds[startId++];
					if (primId == startPrimId)
						continue;
					const i32 sphId = primId - offsetSpheres;
					const ei::Vec4 v = sphVertices[sphId];
					const ei::Sphere sph = { ei::Vec3(v), v.w };
					float t;
					if (ei::intersects(ray, sph, t)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
						}
					}
				}
			}
			// Postponed next node.
			nodeAddr = *(int*)stackPtr;
			stackPtr -= 4;
		}
	}
	(*result) = { hitT, hitPrimId };
#endif // 0

}

void intersection_test_CUDA(const ei::Ray ray, const i32 startPrimId,
	const ei::Vec4* bvh,
	const ei::Vec3* triVertices,
	const ei::Vec3* quadVertices,
	const ei::Vec4* sphVertices,
	const i32* triIndices,
	const i32* quadIndices,
	const i32* primIds,
	const i32 offsetQuads, const i32 offsetSpheres,
	RayIntersectionResult* result,
	const float tmin, const float tmax) {

	intersection_testD << <1, 1 >> > (ray, startPrimId, bvh, triVertices, quadVertices, sphVertices,
		triIndices, quadIndices, primIds, offsetQuads, offsetSpheres, result, tmin, tmax);

}

}
}
}