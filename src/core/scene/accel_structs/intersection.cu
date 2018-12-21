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

template < Device dev >
CUDA_FUNCTION bool intersects_primitve(
	const ObjectDescriptor<dev>& obj,
	const ei::Ray& ray,
	const i32 primId,
	const i32 startPrimId,
	int& hitPrimId,
	float& hitT,				// In out: max hit distance before, if hit then returns the new distance
	ei::Vec3& hitBarycentric
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
			hitBarycentric = barycentric;
			hitPrimId = primId;
			return true;
		}
	} else if(primId < (i32)(obj.polygon.numTriangles + obj.polygon.numQuads)) {
		// Quad.
		const i32 indexOffset = (primId - obj.polygon.numTriangles) * 4 + obj.polygon.numTriangles * 3;
		const ei::Vec3* meshVertices = obj.polygon.vertices;
		float t;
		ei::Vec3 barycentric;

		// Check first triangle
		if(startPrimId != primId) { // Masking to avoid self intersections
			const ei::IVec3 ids = { obj.polygon.vertexIndices[indexOffset],
									obj.polygon.vertexIndices[indexOffset + 1],
									obj.polygon.vertexIndices[indexOffset + 2] };
			const ei::Triangle tri = { meshVertices[ids[0]],
									   meshVertices[ids[1]],
									   meshVertices[ids[2]] };
			if(ei::intersects(ray, tri, t, barycentric) && t < hitT) {
				hitT = t;
				hitBarycentric = barycentric;
				hitPrimId = hitPrimId;
				return true;
			}
		}

		// Check second triangle
		if(startPrimId != (primId | SECOND_QUAD_TRIANGLE_BIT)) { // Masking to avoid self intersections
			const ei::IVec3 ids = { obj.polygon.vertexIndices[indexOffset],
									obj.polygon.vertexIndices[indexOffset + 2],
									obj.polygon.vertexIndices[indexOffset + 3] };
			const ei::Triangle tri = { meshVertices[ids[0]],
									   meshVertices[ids[1]],
									   meshVertices[ids[2]] };
			if(ei::intersects(ray, tri, t, barycentric) && t < hitT) {
				hitT = t;
				hitBarycentric = barycentric;
				hitPrimId = hitPrimId | SECOND_QUAD_TRIANGLE_BIT;
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
			// Barycentrics unused
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
		ei::Vec3 hitBarycentric;
		i32 hitPrimitiveId;
		if(intersects_primitve(obj, ray, 0, startPrimId, hitPrimitiveId, hitT, hitBarycentric)) {
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
			const i32 primId = bvh.primIds[ nodeAddr - bvh.numInternalNodes ];

			// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
			for(i32 i = 0; i < primCount; i++) {
				float hitT = tmax;
				ei::Vec3 hitBarycentric;
				i32 hitPrimitiveId;
				if(intersects_primitve(obj, ray, primId, startPrimId, hitPrimitiveId, hitT, hitBarycentric))
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
	ei::Vec3& hitBarycentric,
	i32* traversalStack
) {
	// Fast path - no BVH
	if(obj.numPrimitives == 1) {
		return intersects_primitve(obj, ray, 0, startPrimId,
			hitPrimId, hitT, hitBarycentric);
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
			const i32 primId = bvh.primIds[ nodeAddr - bvh.numInternalNodes ];

			// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
			for(i32 i = 0; i < primCount; i++) {
				if(intersects_primitve(obj, ray, primId, startPrimId, hitPrimId, hitT, hitBarycentric))
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
	ei::Vec3& hitBarycentric
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
	float t;
	if(intersect(box.min, box.max, invDir, ood, tmin, hitT, t)) {
		// Intersect the ray against the obj primitive bvh.
		const ObjectDescriptor<dev>& obj = scene.objects[objId];
		const LBVH* lbvh = (LBVH*)obj.accelStruct.accelParameters;
		const i32 checkPrimId = (startInsPrimId.instanceId == instanceId) ? startInsPrimId.primId : IGNORE_ID;
		if(first_intersection_obj_lbvh_imp(
			*lbvh, obj, transRay, checkPrimId, invDir, ood, tmin,
			hitPrimId, hitT, hitBarycentric, traversalStack))
			hitInstanceId = instanceId;
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
	i32 hitPrimId = SECOND_QUAD_TRIANGLE_BIT;						// No primitive intersected so far.
	i32 hitInstanceId = IGNORE_ID;
	ei::Vec3 hitBarycentric;
	float hitT = tmax;						// t-value of the closest intersection.

	if(scene.numInstances == 1) {
		i32 traversalStack[OBJ_STACK_SIZE];
		first_intersection_scene_obj_lbvh(
			scene, ray, startInsPrimId, 0, traversalStack,
			hitT, hitInstanceId, hitPrimId, hitBarycentric);
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
				const i32 instanceId = bvh.primIds[ nodeAddr - bvh.numInternalNodes ];

				// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
				for(i32 i = 0; i < primCount; i++) {
					first_intersection_scene_obj_lbvh(scene, ray, startInsPrimId,
						instanceId, stackPtr+1, hitT, hitInstanceId, hitPrimId,
						hitBarycentric);
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
			if(intersects_primitve(obj, transRay, p, -1, hitPrimId, hitT, hitBarycentric))
				hitInstanceId = i;
		}
	}*/

	if(hitInstanceId == IGNORE_ID) {
		return { hitT, { IGNORE_ID, IGNORE_ID } };
	} else {
		// To be determined
		ei::Vec3 normal;
		ei::Vec3 tangent;
		ei::Vec2 uv;

		bool hitSecondTri = false;
		i32 primId = hitPrimId;
		if (hitPrimId & SECOND_QUAD_TRIANGLE_BIT) {
			primId = hitPrimId & SECOND_QUAD_TRIANGLE_MASK;
			hitSecondTri = true;
		}

		const ObjectDescriptor<dev>& obj = scene.objects[ scene.objectIndices[hitInstanceId] ];
		const i32 offsetSpheres = obj.polygon.numTriangles + obj.polygon.numQuads;
		if(primId >= offsetSpheres) { // Sphere?
			const i32 sphId = primId - offsetSpheres;
			const ei::Vec3 hitPoint = ray.origin + hitT * ray.direction;
			normal = normalize(hitPoint - obj.spheres.spheres[sphId].center);

			// Normalization is done later
			if(normal.x == 0.f && normal.y == 0.f)
				tangent = ei::Vec3(1.f, 0.f, 0.f);
			else
				tangent = ei::Vec3(ei::Vec2(normal.y, -normal.x), 0.f);

			uv.x = atan2f(normal.x, normal.y) / (2.f * ei::PI) + 0.5f;
			uv.y = 0.5f * normal.z + 0.5f;
		} else {
			const i32* indices = (i32*)obj.polygon.vertexIndices;
			const ei::Vec3* meshVertices = obj.polygon.vertices;
			const ei::Vec2* meshUVs = obj.polygon.uvs;
			i32 triId;
			if(primId < (i32)obj.polygon.numTriangles) {
				// Triangle.
				triId = primId * 3;
			} else {
				// Quad.
				triId = (primId - obj.polygon.numTriangles) * 4;
				indices += obj.polygon.numTriangles * 3;
				// TODO: compute quad interpolation coordinate
			}
			ei::IVec3 ids = { indices[triId],
							  hitSecondTri ? indices[triId + 3] : indices[triId + 1],
							  indices[triId + 2] };

			ei::Vec3 v[3] = { meshVertices[ids.x], meshVertices[ids.y], meshVertices[ids.z] };
			tangent = v[1] - v[0]; // TODO: fetch the instance instead (issue #44)
			normal = ei::cross(v[0] - v[2], tangent);

			ei::Vec2 uvV[3] = { meshUVs[ids.x], meshUVs[ids.y], meshUVs[ids.z] };
			uv = uvV[0] * hitBarycentric.x + uvV[1] * hitBarycentric.y +
				 uvV[2] * hitBarycentric.z;
		}

		// TODO: enable this for (probably) better code?
		//normal = ei::normalize(ei::transformDir(normal, transforms[hitInstanceId]));
		//tangent = ei::normalize(ei::transformDir(tangent, transforms[hitInstanceId]));
		const ei::Mat3x3 transMatrix = ei::Mat3x3{ scene.transformations[hitInstanceId] };
		normal = ei::normalize(transMatrix * normal);
		tangent = ei::normalize(transMatrix * tangent);

		return { hitT, { hitInstanceId, hitPrimId }, normal, tangent, uv, hitBarycentric };
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
				const i32 instanceId = bvh.primIds[ nodeAddr - bvh.numInternalNodes ];

				// TODO: no loop here! better use only one 'primitive' and wait for the next while iteration
				for(i32 i = 0; i < primCount; i++) {
					if(any_intersection_scene_obj_lbvh(scene, ray, startInsPrimId,
						instanceId, tmax, stackPtr+1))
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
