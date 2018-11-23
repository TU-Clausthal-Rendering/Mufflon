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

__global__
void intersection_testD(const ei::Ray ray, const i32 startPrimId, 
	const ei::Vec4* __restrict__ bvh,
	const ei::Vec3* __restrict__ triVertices,
	const ei::Vec3* __restrict__ quadVertices,
	const ei::Vec4* __restrict__ sphVertices,
	const ei::Vec2* __restrict__ triUVs,
	const ei::Vec2* __restrict__ quadUVs,
	const i32* __restrict__ triIndices,
	const i32* __restrict__ quadIndices,
	const i32* __restrict__ primIds,
	const i32 offsetQuads, const i32 offsetSpheres,
	RayIntersectionResult* __restrict__ result,
	const float tmin, const float tmax) {
	// Setup traversal.
	// Traversal stack in CUDA thread-local memory.
	i32 traversalStack[STACK_SIZE];
	traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

	i32     hitPrimId;                       // Primitive index of the closest intersection, -1 if none.
	ei::Vec3 hitBarycentric;
	float   hitT;                           // t-value of the closest intersection.
	i32 hitSecondTri;                           // For ray-quad intersection.

	// nodeAddr: Non-negative: current internal node; negative: leaf.
	i32     nodeAddr = 0; // Start from the root.  
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
					ei::Vec3 barycentric;
					if (ei::intersects(ray, tri, t, barycentric)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
							hitBarycentric = barycentric;
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
							  quadVertices[quadIndices[quadId + 1]],
							  quadVertices[quadIndices[quadId + 2]],
							  quadVertices[quadIndices[quadId]] };
					const ei::Triangle tri0 = { v[0], v[1], v[2] };
					float t;
					ei::Vec3 barycentric;
					if (ei::intersects(ray, tri0, t, barycentric)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
							hitBarycentric = barycentric;
							hitSecondTri = 0;
						}
					}

					const ei::Triangle tri1 = { v[0], v[2], v[3] };
					if (ei::intersects(ray, tri1, t)) {
						if (t < hitT) {
							hitT = t;
							hitPrimId = primId;
							hitBarycentric = barycentric;
							hitSecondTri = 1;
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
			nodeAddr = *(i32*)stackPtr;
			stackPtr -= 4;
		}
	}
	if (hitPrimId == -1) {
		(*result) = { hitT, hitPrimId };
	}
	else {
		ei::Vec3 normal;
		ei::Vec3 tangent;
		ei::Vec2 uv;

		if (hitPrimId < offsetSpheres) {
			const ei::Vec3* vertices;
			const ei::Vec2* uvs;
			const i32* indices;
			i32 triId;
			if (hitPrimId < offsetQuads) {
				// Triangle.
				triId = hitPrimId * 3;
				indices = triIndices;
				vertices = triVertices;
				uvs = triUVs;
			}
			else {
				// Quad.
				triId = (hitPrimId - offsetQuads) * 4 + hitSecondTri;
				indices = quadIndices;
				vertices = quadVertices;
				uvs = quadUVs;
			}
			ei::IVec3 ids = { indices[triId],
			ids.y = indices[triId + 1],
			ids.z = indices[triId + 2] };

			ei::Vec3 v[3] = { vertices[ids.x], vertices[ids.y], vertices[ids.z] };
			tangent = ei::normalize(v[1] - v[0]);
			normal = ei::cross(ei::normalize(v[0] - v[2]), tangent);

			ei::Vec2 uvV[3] = { uvs[ids.x], uvs[ids.y], uvs[ids.z] };
			uv = uvV[0] * hitBarycentric.x + uvV[1] * hitBarycentric.y + 
				uvV[2] * hitBarycentric.z;
		}
		else {
			// Sphere.
			const i32 sphId = hitPrimId - offsetSpheres;
			ei::Vec3 hitPoint = ray.origin + hitT * ray.direction;
			const ei::Vec4 v = sphVertices[sphId];
			normal = ei::normalize(hitPoint - ei::Vec3(v));

			if (normal.x == 0.f && normal.y == 0.f) {
				tangent = ei::Vec3(1.f, 0.f, 0.f);
			}
			else {
				tangent = ei::Vec3(ei::normalize(ei::Vec2(normal.y, -normal.x)), 0.f);
			}

			uv.x = atan2f(normal.x, normal.y) / (2.f * ei::PI) + 0.5f;
			uv.y = 0.5f * normal.z + 0.5f;
		}

		(*result) = { hitT, hitPrimId, normal, tangent, uv, hitBarycentric};
	}
}

void intersection_test_CUDA(const ei::Ray ray, const i32 startPrimId,
	const ei::Vec4* bvh,
	const ei::Vec3* triVertices,
	const ei::Vec3* quadVertices,
	const ei::Vec4* sphVertices,
	const ei::Vec2* triUVs,
	const ei::Vec2* quadUVs,
	const i32* triIndices,
	const i32* quadIndices,
	const i32* primIds,
	const i32 offsetQuads, const i32 offsetSpheres,
	RayIntersectionResult* result,
	const float tmin, const float tmax) {

	intersection_testD << <1, 1 >> > (ray, startPrimId, bvh, triVertices, quadVertices, sphVertices,
		triUVs, quadUVs, triIndices, quadIndices, primIds, offsetQuads, offsetSpheres, result, tmin, tmax);

}

}
}
}