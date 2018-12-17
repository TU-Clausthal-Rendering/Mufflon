#include "lbvh.hpp"
#include "util/types.hpp"
#include "core/cuda/cu_lib_wrapper.h"
#include "core/math/sfcurves.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/allocator.hpp"
#include "accel_structs_commons.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif // _MSC_VER


//namespace mufflon { namespace {//TODO use anonym?
namespace mufflon {
namespace scene {
namespace accel_struct {


static_assert(MAX_ACCEL_STRUCT_PARAMETER_SIZE >= sizeof(LBVH),
	"Descriptor parameter block to small for this acceleration structure.");

CUDA_FUNCTION void syncthreads() {
#ifdef __CUDA_ARCH__
	__syncthreads();
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION u64 clzll(u64 v) {
#ifdef __CUDA_ARCH__
	return __clzll(v);
#else
#ifdef _MSC_VER
	return __lzcnt64(v);
#else
	return (v == 0) ? 64 : 63 - (u64)log2f((float)v);
#endif // _MSC_VER
#endif // __CUDA_ARCH__
}

CUDA_FUNCTION u32 clz(u32 v) {
#ifdef __CUDA_ARCH__
	return __clz(v);
#else
#ifdef _MSC_VER
	return __lzcnt(v);
#else
	return (v == 0) ? 32 : 31 - (u32)log2f((float)v);
#endif // _MSC_VER
#endif // __CUDA_ARCH__
}


__host__ __device__ ei::Vec3 get_triangle_centroid(ei::Vec3 v0, ei::Vec3 v1, ei::Vec3 v2) {
	ei::Vec3 lo = min(v0, v1, v2);
	ei::Vec3 hi = max(v0, v1, v2);

	return (lo + hi) * 0.5f;
}

__host__ __device__ ei::Vec3 get_quad_centroid(ei::Vec3 v0, ei::Vec3 v1,
	ei::Vec3 v2, ei::Vec3 v3) {
	ei::Vec3 lo = min(v0, v1, v2, v3);
	ei::Vec3 hi = max(v0, v1, v2, v3);

	return (lo + hi) * 0.5f;
}

// Calculates the point morton code using 63 bits.
__forceinline__ __host__ __device__ u32 calculate_morton_code32(ei::Vec3 point)
{
	// Discretize the unit cube into a 21 bit integer
	ei::UVec3 discretized{ ei::clamp(point * 1024.0f, 0.0f, 1023.0f) };

	return math::part_by_two10(discretized[0]) * 4
		+ math::part_by_two10(discretized[1]) * 2
		+ math::part_by_two10(discretized[2]);
}

// Calculates the point morton code using 63 bits.
__forceinline__ __host__ __device__ u64 calculate_morton_code64(ei::Vec3 point)
{
	// Discretize the unit cube into a 21 bit integer
	ei::UVec3 discretized { ei::clamp(point * 2097152.0f, 0.0f, 2097151.0f) };

	return math::part_by_two21(discretized[0]) * 4
		 + math::part_by_two21(discretized[1]) * 2
		 + math::part_by_two21(discretized[2]);
}

__forceinline__ __host__ __device__
ei::Vec3 normalize_position(ei::Vec3 pos, ei::Vec3 lo, ei::Vec3 hi) {
	ei::Vec3 span = hi - lo;
	return ei::Vec3((pos.x - lo.x) / span.x, (pos.y - lo.y) / span.y, (pos.z - lo.z) / span.z);
}

CUDA_FUNCTION void calculate_morton_codes64(
	ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	u64* mortonCodes,
	i32* sortIndices, ei::Vec3 lo, ei::Vec3 hi,
	const i32 offsetQuads, 
	const i32 offsetSpheres, 
	const i32 numPrimitives,
	const i32 idx) {
	ei::Vec3 centroid;
	if (idx >= offsetSpheres) {
		// Calculate Morton codes for spheres.
		i32 sphId = idx - offsetSpheres;
		ei::Vec4 sph = spheres[sphId];
		centroid = ei::Vec3(sph);
	}
	else {
		if (idx >= offsetQuads) {
			// Calculate Morton codes for quads.
			i32 quadId = (idx - offsetQuads) << 2;
			ei::Vec3 v0 = meshVertices[quadIndices[quadId]];
			ei::Vec3 v1 = meshVertices[quadIndices[quadId + 1]];
			ei::Vec3 v2 = meshVertices[quadIndices[quadId + 2]];
			ei::Vec3 v3 = meshVertices[quadIndices[quadId + 3]];
			centroid = get_quad_centroid(v0, v1, v2, v3);
		}
		else {
			// Calculate Morton codes for triangles.
			i32 triId = idx * 3;
			ei::Vec3 v0 = meshVertices[triIndices[triId]];
			ei::Vec3 v1 = meshVertices[triIndices[triId + 1]];
			ei::Vec3 v2 = meshVertices[triIndices[triId + 2]];
			centroid = get_triangle_centroid(v0, v1, v2);
		}
	}
	const ei::Vec3 normalizedPos = normalize_position(centroid, lo, hi);
	const u64 mortonCode = calculate_morton_code64(normalizedPos);
	mortonCodes[idx] = mortonCode;
	sortIndices[idx] = idx;
}

__global__ void calculate_morton_codes64D(
	ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	u64* mortonCodes,
	i32* sortIndices, 
	const ei::Vec3 lo,
	const ei::Vec3 hi,
	const i32 offsetQuads,
	const i32 offsetSpheres,
	const i32 numPrimitives) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numPrimitives)
		return;

	calculate_morton_codes64(meshVertices,
		spheres, triIndices, quadIndices,
		mortonCodes, sortIndices, lo, hi,
		offsetQuads, offsetSpheres, numPrimitives, idx);
}

CUDA_FUNCTION void calculate_morton_codes32(
	ei::Mat3x4* matrices,
	i32* objIds,
	ei::Box* aabbs,
	const ei::Vec3 lo, 
	const ei::Vec3 hi,
	u32* mortonCodes,
	i32* sortIndices,
	const i32 numInstances,
	const i32 idx
) {
	const i32 objId = objIds[idx];
	const ei::Box aabb = ei::transform(aabbs[objId], matrices[idx]);

	const ei::Vec3 normalizedPos = normalize_position(center(aabb), lo, hi);
	mortonCodes[idx] = calculate_morton_code32(normalizedPos);
	sortIndices[idx] = idx;
}

__global__ void calculate_morton_codes32D(
	ei::Mat3x4* matrices,
	i32* objIds,
	ei::Box* aabbs,
	const ei::Vec3 lo, 
	const ei::Vec3 hi,
	u32* mortonCodes,
	i32* sortIndices,
	const i32 numInstances) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numInstances)
		return;

	calculate_morton_codes32(matrices, objIds, aabbs, lo, hi, mortonCodes, sortIndices, numInstances, idx);
}

template<typename T> 
void get_maximum_occupancy(i32 &gridSize, i32 &blockSize, i32 totalThreads, T func, 
						   i32 dynamicSMemSize = 0)
{
	//blockSize;   // The launch configurator returned block size 
	i32 minGridSize; // The minimum grid size needed to achieve the 
	// maximum occupancy for a full device launch 
	//gridSize;    // The actual grid size needed, based on input size 

	cuda::check_error(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, 0));

	if (blockSize != 0)
		// Round up according to array size 
		gridSize = (totalThreads + blockSize - 1) / blockSize;
	else
		gridSize = minGridSize;
}

struct BoundingBoxFunctor {
	__host__ __device__
	i32 operator()(i32 i) const { return sizeof(ei::Vec4) * 2 * i; }
};

template<typename UnaryFunction, typename T>
void get_maximum_occupancy_variable_smem(i32 &gridSize, i32 &blockSize, i32 totalThreads, T func,
	UnaryFunction blockSizeToDynamicSMemSize)
{
	//blockSize;   // The launch configurator returned block size 
	i32 minGridSize; // The minimum grid size needed to achieve the 
	// maximum occupancy for a full device launch 
	//gridSize;    // The actual grid size needed, based on input size 

	cuda::check_error(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, func, blockSizeToDynamicSMemSize, 0));

	if (blockSize != 0)
		// Round up according to array size 
		gridSize = (totalThreads + blockSize - 1) / blockSize;
	else
		gridSize = minGridSize;
}

CUDA_FUNCTION i32 longestCommonPrefix(u64* sortedKeys,
	i32 numberOfElements, i32 index1, i32 index2, u64 key1)
{
	// No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
	// thread per internal node)
	if (index2 < 0 || index2 >= numberOfElements)
	{
		return 0;
	}

	u64 key2 = sortedKeys[index2];

	if (key1 == key2)
	{
		return 64 + (i32)clzll(index1 ^ index2);
	}

	return (i32)clzll(key1 ^ key2);
}

CUDA_FUNCTION i32 longestCommonPrefix(u32* sortedKeys,
	i32 numberOfElements, i32 index1, i32 index2, u32 key1)
{
	// No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
	// thread per internal node)
	if (index2 < 0 || index2 >= numberOfElements)
	{
		return 0;
	}

	u32 key2 = sortedKeys[index2];

	if (key1 == key2)
	{
		return 32 + clz(index1 ^ index2);
	}

	return clz(key1 ^ key2);
}

CUDA_FUNCTION i32 sgn(i32 number)
{
	return (0 < number) - (0 > number);
}

template <typename T> CUDA_FUNCTION void build_lbvh_tree(
	u32 numPrimitives,
	T* sortedKeys,
	i32 *parents,
	const i32 idx
) {
	const T key1 = sortedKeys[idx];

	const i32 lcp1 = longestCommonPrefix(sortedKeys, numPrimitives, idx, idx + 1, key1);
	const i32 lcp2 = longestCommonPrefix(sortedKeys, numPrimitives, idx, idx - 1, key1);

	const i32 direction = sgn((lcp1 - lcp2));

	// Compute upper bound for the length of the range.
	const i32 minLcp = longestCommonPrefix(sortedKeys, numPrimitives, idx, idx - direction, key1);
	i32 lMax = 128;
	while (longestCommonPrefix(sortedKeys, numPrimitives, idx, idx + lMax * direction, key1) >
		minLcp)
	{
		lMax *= 4;
	}

	// Find other end using binary search.
	i32 l = 0;
	i32 t = lMax;
	while (t > 1)
	{
		t = t / 2;
		if (longestCommonPrefix(sortedKeys, numPrimitives, idx, idx + (l + t) * direction, key1) >
			minLcp)
		{
			l += t;
		}
	}
	const i32 j = idx + l * direction;

	// Find the split position using binary search.
	const i32 nodeLcp = longestCommonPrefix(sortedKeys, numPrimitives, idx, j, key1);
	i32 s = 0;
	i32 divisor = 2;
	t = l;
	const i32 maxDivisor = 1 << (32 - clz(l));
	while (divisor <= maxDivisor)
	{
		t = (l + divisor - 1) / divisor;
		if (longestCommonPrefix(sortedKeys, numPrimitives, idx, idx + (s + t) * direction, key1) >
			nodeLcp)
		{
			s += t;
		}
		divisor *= 2;
	}
	const i32 splitPosition = idx + s * direction + min(direction, 0);

	i32 leftIndex;
	i32 rightIndex;

	// Update left child pointer to a leaf.
	if (min(idx, j) == splitPosition)
	{
		// Children is a leaf, add the number of internal nodes to the index.
		leftIndex = splitPosition + numPrimitives - 1;
	}
	else
	{
		leftIndex = splitPosition;
	}

	// Update right child pointer to a leaf.
	if (max(idx, j) == (splitPosition + 1))
	{
		// Children is a leaf, add the number of internal nodes to the index.
		rightIndex = splitPosition + numPrimitives;
	}
	else
	{
		rightIndex = splitPosition + 1;
	}

	// Set parent nodes.
	parents[leftIndex] = ~idx;
	parents[rightIndex] = idx;

	// Set the parent of the root node to -1.
	if (idx == 0)
	{
		parents[0] = 0xEFFFFFFF;
	}
}

// Note: dataIndices is of length numPrimitives.
template <typename T> __global__ void build_lbvh_treeD(
	u32 numPrimitives,
	T* sortedKeys, 
	i32 *parents
)
{
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Check for valid threads.
	if (idx >= (numPrimitives - 1))
	{
		return;
	}

	build_lbvh_tree<T>(numPrimitives, sortedKeys, parents, idx);
}

struct BBCache {
	ei::Box bb;
	float cost;
	i32 primCount;
};
static_assert(sizeof(BBCache) == 8*sizeof(float), "Alignment of BBCache will be broken.");

CUDA_FUNCTION void calculate_bounding_boxes(
	const u32 numPrimitives,
	ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec4 * __restrict__ boundingBoxes, //TODO remove __restricts?
	i32 *sortedIndices,
	i32 * __restrict__ parents,
	i32 *collapseOffsets,
	const i32 offsetQuads,
	const i32 offsetSpheres,
	const float ci, const float ct0, const float ct1, const float ct2,
	i32* counters,
	const i32 idx,
	const i32 firstThreadInBlock = 0,
	const i32 lastThreadInBlock = 0,
	BBCache* sharedBb = nullptr
	)
{
	// If node is to be collapsed, offsets, dataIndices need to be updated.
	bool checkCollapse = true;
	i32 primId = sortedIndices[idx];

	// Calculate leaves bounding box and set primitives count and intersection test cost.
	ei::Box currentBb;
	float cost(ci), costAsLeaf;
	// Some auxilary variables for calculating primitiveCount.
	ei::IVec4 counts = { 0,0,0,1 }; // x: tri; y: quad; z: sphere; w: total.
	ei::IVec4 otherCounts; // x: tri; y: quad; z: sphere; w: total.
	// primitiveCount stores the numbers of each primitives in form:
	// 2 unused bits + 10 bits triangle + 10 bits quads + 10 bits spheres.
	i32 primitiveCount;
	if (primId >= offsetSpheres) {
		// Calculate bounding box for spheres.
		i32 sphId = primId - offsetSpheres;
		ei::Sphere sph = *reinterpret_cast<ei::Sphere*>(spheres + sphId);
		currentBb = ei::Box(sph);
		primitiveCount = 1;
		cost += ct2;
		counts.z = 1;
	}
	else {
		if (primId >= offsetQuads) {
			// Calculate bounding box for quads.
			i32 quadId = (primId - offsetQuads) << 2;
			ei::Vec3 v[4] = { meshVertices[quadIndices[quadId]],
							  meshVertices[quadIndices[quadId + 1]],
							  meshVertices[quadIndices[quadId + 2]],
							  meshVertices[quadIndices[quadId + 3]] };
			currentBb = ei::Box(v, 4);
			primitiveCount = 0x00000400;
			cost += ct1;
			counts.y = 1;
		}
		else {
			// Calculate bounding box for triangles.
			i32 triId = primId * 3;
			ei::Vec3 v[3] = { meshVertices[triIndices[triId]],
							  meshVertices[triIndices[triId + 1]],
							  meshVertices[triIndices[triId + 2]] };
			currentBb = ei::Box(v, 3);
			primitiveCount = 0x00100000;
			cost += ct0;
			counts.x = 1;
		}
	}

	// Store cost and primitiveCount.
	cost *= ei::surface(currentBb);

	// Update node bounding boxes of current node.
#ifdef __CUDA_ARCH__
	sharedBb[threadIdx.x] = { currentBb, cost, primitiveCount };
#endif
	const i32 numInternalNodes = numPrimitives - 1;
	i32 leafIndex = idx + numInternalNodes;
	i32 boxId = leafIndex << 1;
	boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(primitiveCount) };
	boundingBoxes[boxId + 1] = { currentBb.max, cost };

	syncthreads(); // TODO check this.

	// Initialize.
	i32 current = parents[leafIndex];
	bool lastNodeIsLeftChild = false;
	if (current < 0) {
		current = ~current;
		lastNodeIsLeftChild = true;
	}
	i32 lastNode = idx; // Does not need to be initialized, since leaves will not be collapsed
				  // due to positive values of primitiveCount

			// In the counters array, we have stored the id of the thread that processed the other
		// children of this node.
#ifdef __CUDA_ARCH__
	u32 childThreadId = atomicExch(&counters[current], leafIndex);
#else
	i32 childThreadId = counters[current];
	counters[current] = leafIndex;
#endif // __CUDA_ARCH__

	// The first thread to reach a node will just die.
	if (childThreadId == 0xFFFFFFFF)
	{
		return;
	}

	while (true)
	{
		// Fetch bounding boxes and counts information.
		BBCache childInfo;
		i32 anotherChildId = (lastNodeIsLeftChild) ? lastNode + 1 : lastNode - 1;
		if (childThreadId >= numInternalNodes) {
			childThreadId -= numInternalNodes;
			anotherChildId += numInternalNodes;
		}
#ifdef __CUDA_ARCH__
		if (childThreadId >= firstThreadInBlock && childThreadId <= lastThreadInBlock) {
			// If both child nodes were processed by the same block, we can reuse the values
			// cached in shared memory.
			i32 childThreadIdInBlock = childThreadId - firstThreadInBlock;
			childInfo = sharedBb[childThreadIdInBlock];
		}
		else {
			// The children were processed in different blocks, so we have to find out if the one
			// that was not processed by this thread was the left or right one.
			boxId = anotherChildId << 1;
			ei::Vec4 childBbMin = boundingBoxes[boxId];
			ei::Vec4 childBbMax = boundingBoxes[boxId + 1];
			childInfo = BBCache{
				ei::Box{ei::Vec3{childBbMin}, ei::Vec3{childBbMax}},
				childBbMax.w, float_bits_as_int(childBbMin.w)
			};
		}
#else
		boxId = anotherChildId << 1;
		ei::Vec4 childBbMin = boundingBoxes[boxId];
		ei::Vec4 childBbMax = boundingBoxes[boxId + 1];
		childInfo = BBCache{
			ei::Box{ei::Vec3{childBbMin}, ei::Vec3{childBbMax}},
			childBbMax.w, float_bits_as_int(childBbMin.w)
		};
#endif // __CUDA_ARCH__
		syncthreads(); // @todo check.

		currentBb = ei::Box{ currentBb, childInfo.bb };
		if (checkCollapse) {
			// Calculate primitves counts.
			// Set offsets.
			if (primitiveCount < 0) { // Count < 0 means the node should be collapsed.
				primitiveCount &= 0x7FFFFFFF;
				// offset is the number of internal nodes below the child lastNode.
				i32 offset = counts.w - 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[lastNode] = offset;
				else
					// Since lastNode as right child should be collapsed, lastNode + 1
					// must be one child of lastNode if it has more than 2 primitves.
					collapseOffsets[lastNode + 1] = offset;
			}

			extract_prim_counts(childInfo.primCount, otherCounts);

			if (childInfo.primCount < 0) {
				//childInfo.primCount &= 0x7FFFFFFF; // does not needed due to & 0x3FFFFFFF.
				// offset is the number of internal nodes below the other child.
				i32 offset = otherCounts.w - 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[anotherChildId + 1] = offset;
				else
					// Since theOtherChild as right child should be collapsed, lastNode + 1
					// must be one child of theOtherChild if it has more than 2 primitves.
					collapseOffsets[anotherChildId] = offset;
			}

			// Update primtivesCount.
			counts.x += otherCounts.x;
			counts.y += otherCounts.y;
			counts.z += otherCounts.z;

			if (counts.x > 1023 || counts.y > 1023 || counts.z > 1023) {
				checkCollapse = false;
				// Setting cacheMin.w is here to make sure:
				// even if the current node has checkCollapse = false but be killed in 
				// the next round, however, primitiveCount will be read to disable checkCollapse.
				primitiveCount = 0x00000FFF;
			}
			else {
				// & 0x3FFFFFFF is used to avoid intervention of the leftChildMarkBit.
				primitiveCount += (childInfo.primCount & 0x3FFFFFFF);
				// Calculate costs.
				float area = ei::surface(currentBb);
				cost = ci * area + childInfo.cost + cost;
				counts.w += otherCounts.w;// Determine offset.
				costAsLeaf = area * (counts.x * ct0 + counts.y * ct1 + counts.z * ct2);
				if (costAsLeaf < cost) {
					// Collapse.
					primitiveCount |= 0x80000000;
					// Update cost.
					cost = costAsLeaf;
				}
			}
		}

		// Update last processed node
		lastNode = current;

		// Update current node pointer
		current = parents[current];
		// If current == 0, both left/right children are taken as left children
		// for setting offset if needed.
		if (current < 0) {
			current = ~current;
			lastNodeIsLeftChild = true;
		}
		else {
			lastNodeIsLeftChild = false;
		}

		// Update node bounding box of the last node.
		// Put this operation here because we need to 
		// mark the 2. highest bit of cacheMin.w as 1
		// is the lastNode is left child, else mark as 0.
		// This is for simplifying mark_nodesD.
		if (checkCollapse) {
			if (lastNodeIsLeftChild)
				primitiveCount |= 0x40000000;
			else
				primitiveCount &= 0xBFFFFFFF;
		}

#ifdef __CUDA_ARCH__
		sharedBb[threadIdx.x] = BBCache{ currentBb, cost, primitiveCount };
#endif // __CUDA_ARCH__

		boxId = lastNode << 1;
		boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(primitiveCount) };
		boundingBoxes[boxId + 1] = { currentBb.max, cost };

		syncthreads(); //@todo check.

		if (lastNode == 0) {
			// Print the bounding box of the base node.
			printf("root bounding box:\n%f %f %f\n%f %f %f\n",
				currentBb.min.x, currentBb.min.y, currentBb.min.z,
				currentBb.max.x, currentBb.max.y, currentBb.max.z);
			return;
		}

		// In the counters array, we have stored the id of the thread that processed the other
// children of this node.
#ifdef __CUDA_ARCH__
		childThreadId = atomicExch(&counters[current], idx);
#else
		childThreadId = counters[current];
		counters[current] = idx;
#endif // __CUDA_ARCH__

		// The first thread to reach a node will just die.
		if (childThreadId == 0xFFFFFFFF)
		{
			return;
		}
	}
}

__global__ void calculate_bounding_boxesD(
	const u32 numPrimitives,
	ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec4 * __restrict__ boundingBoxes, //TODO remove __restricts?
	i32 *sortedIndices,
	i32 * __restrict__ parents,
	i32 *collapseOffsets,
	const i32 offsetQuads, 
	const i32 offsetSpheres,
	const float ci, const float ct0, const float ct1, const float ct2,
	i32* counters)
{
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	const i32 firstThreadInBlock = blockIdx.x * blockDim.x;
	const i32 lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

	// Initialize cache of bounding boxes in shared memory.
	extern __shared__ BBCache sharedBb[]; 

	// Check for valid threads.
	if (idx >= numPrimitives)
	{
		return;
	}

	calculate_bounding_boxes(numPrimitives, meshVertices, spheres, triIndices, quadIndices,
		boundingBoxes, sortedIndices, parents, collapseOffsets, offsetQuads, offsetSpheres,
		ci, ct0, ct1, ct2, counters, idx, firstThreadInBlock, lastThreadInBlock, sharedBb);
}

CUDA_FUNCTION void calculate_bounding_boxes_ins(
	const u32 numInstances,
	ei::Mat3x4* matrices,
	i32* objIds,
	ei::Box* aabbs,
	ei::Vec4 * __restrict__ boundingBoxes, //TODO remove __restricts?
	i32 *sortedIndices,
	i32 * __restrict__ parents,
	i32 *collapseOffsets,
	const float ci, 
	const float ct,
	i32* counters,
	const i32 idx,
	const i32 firstThreadInBlock = 0,
	const i32 lastThreadInBlock = 0,
	BBCache* sharedBb = nullptr) {
	// If node is to be collapsed, offsets, dataIndices need to be updated.
	bool checkCollapse = true;

	// Calculate leaves bounding box and set primitives count and intersection test cost.
	float costAsLeaf;
	const i32 objId = objIds[idx];
	ei::Box currentBb = ei::transform(aabbs[objId], matrices[idx]);
	// Some auxilary variables for calculating primitiveCount.
	i32 instanceCount = 1;
	float cost = ci + ct;

	// Store cost and primitiveCount.
	cost *= ei::surface(currentBb);

	// Update node bounding boxes of current node.
#ifdef __CUDA_ARCH__
	sharedBb[threadIdx.x] = { currentBb, cost, instanceCount };
#endif
	const i32 numInternalNodes = numInstances - 1;
	i32 leafIndex = idx + numInternalNodes;
	i32 boxId = leafIndex << 1;
	boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(instanceCount) };
	boundingBoxes[boxId + 1] = { currentBb.max, cost };

	syncthreads(); // TODO check this.

	// Initialize.
	i32 current = parents[leafIndex];
	bool lastNodeIsLeftChild = false;
	if (current < 0) {
		current = ~current;
		lastNodeIsLeftChild = true;
	}
	i32 lastNode = idx; // Does not need to be initialized, since leaves will not be collapsed
				  // due to positive values of primitiveCount

			// In the counters array, we have stored the id of the thread that processed the other
		// children of this node.
#ifdef __CUDA_ARCH__
	u32 childThreadId = atomicExch(&counters[current], leafIndex);
#else
	i32 childThreadId = counters[current]; 
	counters[current] = leafIndex;
#endif // __CUDA_ARCH__

	// The first thread to reach a node will just die.
	if (childThreadId == 0xFFFFFFFF)
	{
		return;
	}

	while (true)
	{
		// Fetch bounding boxes and counts information.
		BBCache childInfo;
		i32 anotherChildId = (lastNodeIsLeftChild) ? lastNode + 1 : lastNode - 1;
		if (childThreadId >= numInternalNodes) {
			childThreadId -= numInternalNodes;
			anotherChildId += numInternalNodes;
		}
#ifdef __CUDA_ARCH__
		if (childThreadId >= firstThreadInBlock && childThreadId <= lastThreadInBlock) {
			// If both child nodes were processed by the same block, we can reuse the values
			// cached in shared memory.
			i32 childThreadIdInBlock = childThreadId - firstThreadInBlock;
			childInfo = sharedBb[childThreadIdInBlock];
		}
		else {
			// The children were processed in different blocks, so we have to find out if the one
			// that was not processed by this thread was the left or right one.
			boxId = anotherChildId << 1;
			ei::Vec4 childBbMin = boundingBoxes[boxId];
			ei::Vec4 childBbMax = boundingBoxes[boxId + 1];
			childInfo = BBCache{
				ei::Box{ei::Vec3{childBbMin}, ei::Vec3{childBbMax}},
				childBbMax.w, float_bits_as_int(childBbMin.w)
			};
		}
#else
		boxId = anotherChildId << 1;
		ei::Vec4 childBbMin = boundingBoxes[boxId];
		ei::Vec4 childBbMax = boundingBoxes[boxId + 1];
		childInfo = BBCache{
			ei::Box{ei::Vec3{childBbMin}, ei::Vec3{childBbMax}},
			childBbMax.w, float_bits_as_int(childBbMin.w)
		};
#endif // __CUDA_ARCH__
		syncthreads(); // @todo check.
		
		currentBb = ei::Box{ currentBb, childInfo.bb };
		if (checkCollapse) {
			// Calculate primitves counts.
			// Set offsets.
			if (instanceCount < 0) { // Count < 0 means the node should be collapsed.
				instanceCount &= 0x3FFFFFFF;
				// offset is the number of internal nodes below the child lastNode.
				i32 offset = instanceCount - 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[lastNode] = offset;
				else
					// Since lastNode as right child should be collapsed, lastNode + 1
					// must be one child of lastNode if it has more than 2 primitves.
					collapseOffsets[lastNode + 1] = offset;
			}

			if (childInfo.primCount < 0) {
				// & 0x3FFFFFFF is used to avoid intervention of the leftChildMarkBit.
				childInfo.primCount &= 0x3FFFFFFF;
				// offset is the number of internal nodes below the other child.
				i32 offset = childInfo.primCount - 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[anotherChildId + 1] = offset;
				else
					// Since theOtherChild as right child should be collapsed, lastNode + 1
					// must be one child of theOtherChild if it has more than 2 primitves.
					collapseOffsets[anotherChildId] = offset;
			}

			// Update primtivesCount.
			instanceCount += childInfo.primCount;
			instanceCount &= 0x3FFFFFFF;

			if (instanceCount > 1023) {
				checkCollapse = false;
				// Setting cacheMin.w is here to make sure:
				// even if the current node has checkCollapse = false but be killed in 
				// the next round, however, primitiveCount will be read to disable checkCollapse.
				instanceCount = 0x00000FFF;
			}
			else {
				// Calculate costs.
				float area = ei::surface(currentBb);
				cost = ci * area + childInfo.cost + cost;
				costAsLeaf = area * instanceCount * ct;
				if (costAsLeaf < cost) {
					// Collapse.
					instanceCount |= 0x80000000;
					// Update cost.
					cost = costAsLeaf;
				}
			}
		}

		// Update last processed node
		lastNode = current;

		// Update current node pointer
		current = parents[current];
		// If current == 0, both left/right children are taken as left children
		// for setting offset if needed.
		if (current < 0) {
			current = ~current;
			lastNodeIsLeftChild = true;
		}
		else {
			lastNodeIsLeftChild = false;
		}

		// Update node bounding box of the last node.
		// Put this operation here because we need to 
		// mark the 2. highest bit of cacheMin.w as 1
		// is the lastNode is left child, else mark as 0.
		// This is for simplifying mark_nodesD.
		if (checkCollapse) {
			if (lastNodeIsLeftChild)
				instanceCount |= 0x40000000;
			else
				instanceCount &= 0xBFFFFFFF;
		}

#ifdef __CUDA_ARCH__
		sharedBb[threadIdx.x] = BBCache{ currentBb, cost, instanceCount };
#endif // __CUDA_ARCH__

		boxId = lastNode << 1;
		boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(instanceCount) };
		boundingBoxes[boxId + 1] = { currentBb.max, cost };

		syncthreads(); //@todo check.

		if (lastNode == 0) {
			// Print the bounding box of the base node.
			printf("root bounding box:\n%f %f %f\n%f %f %f\n",
				currentBb.min.x, currentBb.min.y, currentBb.min.z,
				currentBb.max.x, currentBb.max.y, currentBb.max.z);
			return;
		}

		// In the counters array, we have stored the id of the thread that processed the other
// children of this node.
#ifdef __CUDA_ARCH__
		childThreadId = atomicExch(&counters[current], idx);
#else
		childThreadId = counters[current];
		counters[current] = idx;
#endif // __CUDA_ARCH__
		

		// The first thread to reach a node will just die.
		if (childThreadId == 0xFFFFFFFF)
		{
			return;
		}
	}
}

__global__ void calculate_bounding_boxes_insD(
	u32 numInstances,
	ei::Mat3x4* matrices,
	i32* objIds,
	ei::Box* aabbs,
	ei::Vec4 * __restrict__ boundingBoxes, //TODO remove __restricts?
	i32 *sortedIndices,
	i32 * __restrict__ parents,
	i32 *collapseOffsets,
	float ci, float ct,
	i32* counters)
{
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	const i32 firstThreadInBlock = blockIdx.x * blockDim.x;
	const i32 lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

	// Initialize cache of bounding boxes in shared memory.
	extern __shared__ BBCache sharedBb[];

	// Check for valid threads.
	if (idx >= numInstances)
	{
		return;
	}

	calculate_bounding_boxes_ins(numInstances, matrices, objIds, aabbs, boundingBoxes,
		sortedIndices, parents, collapseOffsets, ci, ct, counters, idx, 
		firstThreadInBlock, lastThreadInBlock, sharedBb);
}

CUDA_FUNCTION void mark_nodes(
	u32 numInternalNodes,
	ei::Vec4* __restrict__ boundingBoxes,
	i32* __restrict__ removedMarks,
	i32* __restrict__ reduceOffsets,
	i32* __restrict__ leafMarks,
	const i32 idx
) {
	// This is better than using a stack, since less operations are needed
	// and due to parallerism, performance is not affected.
	const i32 boxId = idx << 1;
	i32 primitiveCount = float_bits_as_int(boundingBoxes[boxId].w);
	if (primitiveCount >= 0) {
		leafMarks[idx] = 0;
		return;
	}
	leafMarks[idx] = 1;

	ei::IVec4 count;
	// Extract counts for three kinds of primitvies.
	extract_prim_counts(primitiveCount, count);
	//if (count.w == 2) // Not needed due to start <= end check.
	//	return; 
	i32 start, end;
	if ((primitiveCount & 0x40000000) == 0) {
		// Current node is a right child.
		start = idx + 1;
		end = idx + count.w - 2;
	}
	else {
		// Current node is a left child.
		end = idx - 1;
		start = idx - count.w + 2;
	}
	while (start <= end) {
		removedMarks[start] = 0xFFFFFFFF;
		reduceOffsets[start] = 0;
		++start;
	}
}

__global__ void mark_nodesD(
	u32 numInternalNodes,
	ei::Vec4* __restrict__ boundingBoxes,
	i32* __restrict__ removedMarks,
	i32* __restrict__ reduceOffsets,
	i32* __restrict__ leafMarks
) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numInternalNodes)
		return;

	mark_nodes(numInternalNodes, boundingBoxes, removedMarks, reduceOffsets, leafMarks, idx);
}


CUDA_FUNCTION i32 insert_id(i32 id, i32* preLeaves, i32 removedLeaves) {
	i32 numPreLeaves = preLeaves[id];
	return (id << 2) - numPreLeaves * 3 - removedLeaves;
}

CUDA_FUNCTION i32 next_id(ei::IVec4& insertPos, ei::IVec2& endPos, i32& primType) {
	if (insertPos.x < endPos.x) {
		primType = 0;
		return insertPos.x;
	}
	else if (insertPos.y < endPos.y) {
		primType = 1;
		return insertPos.y;
	}
	else {
		primType = 2;
		return insertPos.z;
	}
}

CUDA_FUNCTION void copy_to_collapsed_bvh(
	const i32 numNodes,
	const i32 numInternalNodes,
	const i32 simpleLeafOffset,
	ei::Vec4* collapsedBVH,
	ei::Vec4 * __restrict__ boundingBoxes,
	i32* __restrict__ parents,
	i32* __restrict__ removedMarks,
	i32* __restrict__ sortIndices,
	i32* __restrict__ preInternalLeaves,
	i32* __restrict__ reduceOffsets,
	const i32 offsetQuads, const i32 offsetSpheres,
	const i32 idx
) {
	if (idx >= numInternalNodes) {
		i32 parent = parents[idx];
		i32 offset = 1;
		if (parent < 0) {// Is left child.
			parent = ~parent;
			offset = 0;
		}

		const i32 boxId = parent << 1;
		i32 primitiveCount = float_bits_as_int(boundingBoxes[boxId].w);
		if (primitiveCount >= 0 && removedMarks[parent] != 0xFFFFFFFF) {

			// Copy bounding boxes to parent.
			// Set to parent id in collapsed bvh.
			const i32 removedLeaves = reduceOffsets[parent];
			i32 nodeId = idx << 1;
			ei::Vec4 lo = boundingBoxes[nodeId];
			ei::Vec4 hi = boundingBoxes[nodeId + 1];
			const i32 insertId = insert_id(parent, preInternalLeaves, removedLeaves);
			collapsedBVH[insertId + offset] = ei::Vec4(lo.x, hi.x, lo.y, hi.y);
			*(((ei::Vec2*)&collapsedBVH[insertId + 2]) + offset) = ei::Vec2(lo.z, hi.z);

			// Set child pointers to parent.
			// set to current node id in collapsed bvh.
			*(((i32*)&collapsedBVH[insertId + 3]) + offset) =
				~(simpleLeafOffset + idx);
		}

	} else
	if (removedMarks[idx] != 0xFFFFFFFF) {
		i32 parent = parents[idx];
		i32 offset = 1;
		if (parent < 0) {// Is left child.
			parent = ~parent;
			offset = 0;
		}

		// Copy bounding boxes to parent.
		// Set to parent id in collapsed bvh.
		i32 removedInternalLeaves = reduceOffsets[parent];
		i32 nodeId = idx << 1;
		const ei::Vec4 lo = boundingBoxes[nodeId];
		const ei::Vec4 hi = boundingBoxes[nodeId + 1];
		const i32 insertId = insert_id(parent, preInternalLeaves, removedInternalLeaves);
		collapsedBVH[insertId + offset] = ei::Vec4(lo.x, hi.x, lo.y, hi.y);
		*(((ei::Vec2*)&collapsedBVH[insertId + 2]) + offset) = ei::Vec2(lo.z, hi.z);

		// Set child pointers to parent.
		// set to current node id in collapsed bvh.
		removedInternalLeaves = reduceOffsets[idx];
		int pointerId = insert_id(idx, preInternalLeaves, removedInternalLeaves);
		// Leaf with negative sign to the pointer.
		*(((i32*)&collapsedBVH[insertId + 3]) + offset) = (lo.w < 0) ? ~pointerId: pointerId;
		
		// Copy data indices.
		if (lo.w < 0) {
			i32 primitiveCount = float_bits_as_int(lo.w);
			ei::IVec4 counts;
			extract_prim_counts(primitiveCount, counts);
			i32 startId;
			if (offset) { // The current node is right child.
				startId = idx;
			}
			else {
				startId = idx - counts.w + 1;
			}

			// Now counts is used to register insert position.
			ei::IVec2 endPos;
			endPos.x = startId + counts.x;
			endPos.y = endPos.x + counts.y;
			i32 primId;
			i32 nextId;
			i32 primInsertId;
			bool readId = true;
			counts.x = startId;
			counts.y = endPos.x;
			counts.z = endPos.y;
			collapsedBVH[pointerId] = ei::Vec4(
				lo.w, int_bits_as_float(counts.x),
				int_bits_as_float(counts.y), int_bits_as_float(counts.z));
			i32 primType; // 0: tri; 1: quad; 2: sph.
			i32 tmpId;
			// Use a loop to set data indices.
			// TODO: try simple read/write version to reduce branches.
			while (counts.w != 0) {
				if (readId) {
					nextId = next_id(counts, endPos, primType);
					primId = sortIndices[nextId];

					if (primId < offsetQuads) {
						// Read a triangle.
						if (primType == 0) {
							goto stay_at_place;
						}
						primType = 0;
					}
					else if (primId < offsetSpheres) {
						// Read a quad.
						if (primType == 1) {
							goto stay_at_place;
						}
						primType = 1;
					}
					else { // Read a sphere.
						if (primType == 2) {
							goto stay_at_place;
						}
						primType = 2;
					}
					readId = false;
				}

				//Now primType matches current primId.
				primInsertId = counts[primType]++;
				--counts.w;

				if ((primInsertId == nextId)) {
					readId = true;
				}
				else {
					tmpId = sortIndices[primInsertId];
					if (tmpId < offsetQuads) {
						if (primType == 0) {
							continue;
						}
						primType = 0;
					}
					else if (tmpId < offsetSpheres) {
						if (primType == 1) {
							continue;
						}
						primType = 1;
					}
					else if (primType == 2) {
						continue;
					}
					else {
						primType = 2;
					}

				}
				sortIndices[primInsertId] = primId;
				primId = tmpId;
				continue;

			stay_at_place:
				--counts.w; // Reduce the numbe of to be inserted primitives.
				counts[primType]++;
			}
		}
	}
}

__global__ void copy_to_collapsed_bvhD(
	const i32 numNodes,
	const i32 numInternalNodes,
	const i32 simpleLeafOffset,
	ei::Vec4* collapsedBVH,
	ei::Vec4 * __restrict__ boundingBoxes,
	i32* __restrict__ parents,
	i32* __restrict__ removedMarks,
	i32* __restrict__ sortIndices,
	i32* __restrict__ preInternalLeaves,
	i32* __restrict__ reduceOffsets,
	const i32 offsetQuads, const i32 offsetSpheres
) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x + 1;

	if (idx >= numNodes)
		return;

	copy_to_collapsed_bvh(numNodes, numInternalNodes, simpleLeafOffset, collapsedBVH,
		boundingBoxes, parents, removedMarks, sortIndices, preInternalLeaves, reduceOffsets,
		offsetQuads, offsetSpheres, idx);
}

CUDA_FUNCTION void copy_to_collapsed_bvh_ins(
	i32 numNodes,
	i32 numInternalNodes,
	i32 simpleLeafOffset,
	ei::Vec4* collapsedBVH,
	ei::Vec4 * __restrict__ boundingBoxes,
	i32* __restrict__ parents,
	i32* __restrict__ removedMarks,
	i32* __restrict__ sortIndices,
	i32* __restrict__ preInternalLeaves,
	i32* __restrict__ reduceOffsets,
	const i32 idx
) {
	if (idx >= numInternalNodes) {
		i32 parent = parents[idx];
		i32 offset = 1;
		if (parent < 0) {// Is left child.
			parent = ~parent;
			offset = 0;
		}

		const i32 boxId = parent << 1;
		i32 primitiveCount = float_bits_as_int(boundingBoxes[boxId].w);
		if (primitiveCount >= 0 && removedMarks[parent] != 0xFFFFFFFF) {

			// Copy bounding boxes to parent.
			// Set to parent id in collapsed bvh.
			const i32 removedLeaves = reduceOffsets[parent];
			i32 nodeId = idx << 1;
			ei::Vec4 lo = boundingBoxes[nodeId];
			ei::Vec4 hi = boundingBoxes[nodeId + 1];
			const i32 insertId = insert_id(parent, preInternalLeaves, removedLeaves);
			collapsedBVH[insertId + offset] = ei::Vec4(lo.x, hi.x, lo.y, hi.y);
			*(((ei::Vec2*)&collapsedBVH[insertId + 2]) + offset) = ei::Vec2(lo.z, hi.z);

			// Set child pointers to parent.
			// set to current node id in collapsed bvh.
			*(((i32*)&collapsedBVH[insertId + 3]) + offset) =
				~(simpleLeafOffset + idx);
		}

	}
	else
		if (removedMarks[idx] != 0xFFFFFFFF) {
			i32 parent = parents[idx];
			i32 offset = 1;
			if (parent < 0) {// Is left child.
				parent = ~parent;
				offset = 0;
			}

			// Copy bounding boxes to parent.
			// Set to parent id in collapsed bvh.
			i32 removedInternalLeaves = reduceOffsets[parent];
			i32 nodeId = idx << 1;
			const ei::Vec4 lo = boundingBoxes[nodeId];
			const ei::Vec4 hi = boundingBoxes[nodeId + 1];
			const i32 insertId = insert_id(parent, preInternalLeaves, removedInternalLeaves);
			collapsedBVH[insertId + offset] = ei::Vec4(lo.x, hi.x, lo.y, hi.y);
			*(((ei::Vec2*)&collapsedBVH[insertId + 2]) + offset) = ei::Vec2(lo.z, hi.z);

			// Set child pointers to parent.
			// set to current node id in collapsed bvh.
			removedInternalLeaves = reduceOffsets[idx];
			int pointerId = insert_id(idx, preInternalLeaves, removedInternalLeaves);
			// Leaf with negative sign to the pointer.
			*(((i32*)&collapsedBVH[insertId + 3]) + offset) = (lo.w < 0) ? ~pointerId : pointerId;

			// Set the pointer to instances.
			// Copy data indices is no longer needed.
			if (lo.w < 0) {
				i32 instanceCount = float_bits_as_int(lo.w) & 0x3FFFFFFF;
				i32 startId;
				if (offset) { // The current node is right child.
					startId = idx;
				}
				else {
					startId = idx - instanceCount + 1;
				}
				collapsedBVH[pointerId] = ei::Vec4(
					instanceCount, int_bits_as_float(startId), 0.f, 0.f);
			}
		}
}

__global__ void copy_to_collapsed_bvh_insD(
	i32 numNodes,
	i32 numInternalNodes,
	i32 simpleLeafOffset,
	ei::Vec4* collapsedBVH,
	ei::Vec4 * __restrict__ boundingBoxes,
	i32* __restrict__ parents,
	i32* __restrict__ removedMarks,
	i32* __restrict__ sortIndices,
	i32* __restrict__ preInternalLeaves,
	i32* __restrict__ reduceOffsets
) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x + 1;

	if (idx >= numNodes)
		return;


	copy_to_collapsed_bvh_ins(numNodes, numInternalNodes, simpleLeafOffset, collapsedBVH,
		boundingBoxes, parents, removedMarks, sortIndices, preInternalLeaves, reduceOffsets, idx);
}

//}} // namespace mufflon:: {
}}}


namespace mufflon { namespace scene { namespace accel_struct {

// For the objects.
template < Device dev >
void LBVHBuilder::build_lbvh64(ei::Vec3* meshVertices,
	ei::Vec4* spheres,
	i32* triIndices,
	i32* quadIndices,
	const ei::Box& aabb, ei::Vec4 traverseCosts, i32 numPrimitives,
	i32 offsetQuads, i32 offsetSpheres) {
	if (numPrimitives == 1) {
		// TODO remove this. 
		m_primIds.resize(1);
		m_primIds.synchronize<dev>();
		m_bvhNodes.resize(1);
		m_bvhNodes.synchronize<dev>();
		return;
	}

	i32 numBlocks, numThreads;

	// Allocate memory for a part of the BVH.We do not know the final size yet and
	// cannot allocate the other parts in bvh.
	m_primIds.resize(numPrimitives * sizeof(i32));
	i32* primIds = as<i32>(m_primIds.acquire<dev>());

	// Calculate Morton codes.
	auto codesMem = make_udevptr_array<dev, u64>(numPrimitives);
	const i32 numInternalNodes = numPrimitives - 1;
	const u32 numNodes = numInternalNodes + numPrimitives;
	auto parentsMem = make_udevptr_array<dev, i32>(numNodes);
	i32 *parents = parentsMem.get(); // size numNodes.
	{
		u64* mortonCodes = codesMem.get();
		if (dev == Device::CUDA) {
			get_maximum_occupancy(numBlocks, numThreads, numPrimitives, calculate_morton_codes64D);
			calculate_morton_codes64D << < numBlocks, numThreads >> > (meshVertices,
				spheres, triIndices, quadIndices,
				mortonCodes, primIds, aabb.min, aabb.max,
				offsetQuads, offsetSpheres, numPrimitives);

			// Sort based on Morton codes.
			CuLib::DeviceSort(numPrimitives, &mortonCodes, &mortonCodes,
				&primIds, &primIds);
		}
		else {
			for (i32 idx = 0; idx < numPrimitives; idx++)
			{
				calculate_morton_codes64(meshVertices,
					spheres, triIndices, quadIndices,
					mortonCodes, primIds, aabb.min, aabb.max,
					offsetQuads, offsetSpheres, numPrimitives, idx);
			}

			// Sort based on Morton codes.
			thrust::sort_by_key(mortonCodes, mortonCodes + numPrimitives, primIds);
		}

		// Create BVH.
		// Layout: first internal nodes, then leves.
		if (dev == Device::CUDA) {
			cudaFuncSetCacheConfig(build_lbvh_treeD<u64>, cudaFuncCachePreferL1);
			get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, build_lbvh_treeD<u64>);
			build_lbvh_treeD<u64> << <numBlocks, numThreads >> > (//TODO check <u64>
				numPrimitives,
				mortonCodes,
				parents);
		}
		else {
			for (i32 idx = 0; idx < numInternalNodes; idx++)
			{
				build_lbvh_tree<u64>(numPrimitives, mortonCodes, parents, idx);
			}
		}
	}

	// Calcualte bounding boxes and SAH.
	// Create atomic counters buffer.
	auto deviceCountersMem = make_udevptr_array<dev, i32>(numInternalNodes);
	i32* deviceCounters = deviceCountersMem.get();
	mem_set<dev>(deviceCounters, 0xFF, numInternalNodes * sizeof(i32));
	// Allocate bounding boxes.
	auto boundingBoxesMem = make_udevptr_array<dev, ei::Vec4>(numNodes * 2);
	ei::Vec4 *boundingBoxes = boundingBoxesMem.get();
	// Allocate collapseOffsets.
	// The last position for collapseOffsets is to avoid access violations,
	// since if the last internal node needs to be collapsed, it will write 
	// to this positions with offset = 0, but this info will not be used further.
	auto collapseOffsetsMem = make_udevptr_array<dev, i32>(numInternalNodes);
	i32* collapseOffsets = collapseOffsetsMem.get();
	mem_set<dev>(collapseOffsets, 0, numInternalNodes * sizeof(i32));
	i32* leafMarks = (i32*)codesMem.get();

	if (dev == Device::CUDA) {
		// Calculate BVH bounding boxes.
		i32 bboxCacheSize = numThreads * sizeof(ei::Vec4) * 2;
		cudaFuncSetCacheConfig(calculate_bounding_boxesD, cudaFuncCachePreferShared);
		BoundingBoxFunctor functor;
		get_maximum_occupancy_variable_smem(numBlocks, numThreads, numPrimitives,
			calculate_bounding_boxesD, functor);
		calculate_bounding_boxesD << <numBlocks, numThreads, bboxCacheSize >> > (
			numPrimitives,
			meshVertices,
			spheres, triIndices, quadIndices,
			boundingBoxes,
			primIds,
			parents,
			collapseOffsets,
			offsetQuads, offsetSpheres,
			traverseCosts.x, traverseCosts.y, traverseCosts.z, traverseCosts.w,
			deviceCounters);

		// Mark all children of collapsed nodes as removed and themselves as leaves (=1).
		get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, mark_nodesD);
		mark_nodesD << <numBlocks, numThreads, bboxCacheSize >> > (
			numInternalNodes,
			boundingBoxes,
			deviceCounters,
			collapseOffsets,
			leafMarks);
	}
	else {
		// Calculate BVH bounding boxes.
		for (i32 idx = 0; idx < numPrimitives; idx++)
		{
			calculate_bounding_boxes(numPrimitives, meshVertices, spheres, triIndices, quadIndices,
				boundingBoxes, primIds, parents, collapseOffsets, offsetQuads, offsetSpheres,
				traverseCosts.x, traverseCosts.y, traverseCosts.z, traverseCosts.w, deviceCounters, idx);
		}

		// Mark all children of collapsed nodes as removed and themselves as leaves (=1).
		for (i32 idx = 0; idx < numInternalNodes; idx++)
		{
			mark_nodes(numInternalNodes, boundingBoxes, deviceCounters, collapseOffsets, leafMarks, idx);
		}
	}

	i32 numRemovedInternalNodes;
	if (dev == Device::CUDA) {
		// Scan to get values for offsets.
		CuLib::DeviceInclusiveSum(numInternalNodes, &collapseOffsets, &collapseOffsets);
		//cudaDeviceSynchronize();//testing
		cudaMemcpy(&numRemovedInternalNodes, collapseOffsets + numInternalNodes - 1, sizeof(i32),
			cudaMemcpyDeviceToHost);
		// Scan to get number of leaves arised from internal nodes before current node.
		CuLib::DeviceExclusiveSum(numInternalNodes + 1, &leafMarks, &leafMarks);
	}
	else {
		// Scan to get values for offsets.
		thrust::inclusive_scan(collapseOffsets, collapseOffsets + numInternalNodes,
			collapseOffsets);
		numRemovedInternalNodes = collapseOffsets[numInternalNodes - 1];
		// Scan to get number of leaves arised from internal nodes before current node.
		thrust::exclusive_scan(leafMarks, leafMarks + numInternalNodes + 1, leafMarks);
	}

	// Here uses a compact memory layout so that each leaf only uses 16 bytes.
	i32 numInternLeavesCollapsedBVH; //!= (numNodesCollapsedBVH + 1) >> 1; since there are simple leaves.
	// and it includes the removed ones.

	if (dev == Device::CUDA)
		cudaMemcpy(&numInternLeavesCollapsedBVH, leafMarks + numInternalNodes, sizeof(i32),
			cudaMemcpyDeviceToHost);
	else
		numInternLeavesCollapsedBVH = leafMarks[numInternalNodes];

	i32 numInternalNodesCollapsedBVH = numInternalNodes - numInternLeavesCollapsedBVH;
	numInternLeavesCollapsedBVH -= numRemovedInternalNodes;
	i32 numFloat4InCollapsedBVH = numInternLeavesCollapsedBVH + 4 * numInternalNodesCollapsedBVH;
	//printf("bvhSize %d %d %d\n", bvhSize, 
	//	numInternLeavesCollapsedBVH + numRemovedInternalNodes, numRemovedInternalNodes);
	m_bvhNodes.resize(numFloat4InCollapsedBVH * sizeof(ei::Vec4));
	ei::Vec4* collapsedBVH = as<ei::Vec4>(m_bvhNodes.acquire<dev>());
	if (dev == Device::CUDA) {
		get_maximum_occupancy(numBlocks, numThreads, numNodes, copy_to_collapsed_bvhD);
		copy_to_collapsed_bvhD << < numBlocks, numThreads >> > (
			numNodes,
			numInternalNodes,
			numFloat4InCollapsedBVH - numInternalNodes,
			collapsedBVH,
			boundingBoxes,
			parents,
			deviceCounters,
			primIds,
			leafMarks,
			collapseOffsets,
			offsetQuads, offsetSpheres
			);
	}
	else {
		for (i32 idx = 1; idx < i32(numNodes); idx++)
		{
			copy_to_collapsed_bvh(numNodes, numInternalNodes, numFloat4InCollapsedBVH - numInternalNodes,
				collapsedBVH, boundingBoxes, parents, deviceCounters, primIds,
				leafMarks, collapseOffsets, offsetQuads, offsetSpheres, idx);
		}
	}
}

// For the scene.
template < Device dev >
void LBVHBuilder::build_lbvh32(ei::Mat3x4* matrices,
	i32* objIds,
	ei::Box* aabbs,
	const ei::Box& sceneBB,
	ei::Vec2 traverseCosts, i32 numInstances) {
	if (numInstances == 1) {
		// TODO remove this. 
		m_primIds.resize(1);
		m_primIds.synchronize<dev>();
		m_bvhNodes.resize(1);
		m_bvhNodes.synchronize<dev>();
		return;
	}

	i32 numBlocks, numThreads;

	// Allocate memory for a part of the BVH. We do not know the final size yet and
	// cannot allocate the other parts in bvh.
	m_primIds.resize(numInstances * sizeof(i32));
	i32* primIds = as<i32>(m_primIds.acquire<dev>());

	// Calculate Morton codes.
	auto codesMem = make_udevptr_array<dev, u32>(numInstances);
	const i32 numInternalNodes = numInstances - 1;
	const u32 numNodes = numInternalNodes + numInstances;
	auto parentsMem = make_udevptr_array<dev, i32>(numNodes);
	i32 *parents = parentsMem.get(); // size numNodes.

	{
		u32* mortonCodes = codesMem.get();
		if (dev == Device::CUDA) {
			get_maximum_occupancy(numBlocks, numThreads, numInstances, calculate_morton_codes32D);
			calculate_morton_codes32D << < numBlocks, numThreads >> > (
				matrices, objIds, aabbs, sceneBB.min, sceneBB.max,
				mortonCodes, primIds, numInstances);

			// Sort based on Morton codes.
			CuLib::DeviceSort(numInstances, &mortonCodes, &mortonCodes,
				&primIds, &primIds);
		}
		else {
			for (i32 idx = 0; idx < numInstances; idx++)
			{
				calculate_morton_codes32(matrices, objIds, aabbs, sceneBB.min, sceneBB.max,
					mortonCodes, primIds, numInstances, idx);
			}

			// Sort based on Morton codes.
			thrust::sort_by_key(mortonCodes, mortonCodes + numInstances, primIds);
		}

		// Create BVH.
		// Layout: first internal nodes, then leves.
		if (dev == Device::CUDA) {
			cudaFuncSetCacheConfig(build_lbvh_treeD<u64>, cudaFuncCachePreferL1);
			get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, build_lbvh_treeD<u32>);
			build_lbvh_treeD<u32> << <numBlocks, numThreads >> > (//TODO check <u64>
				numInstances,
				mortonCodes,
				parents);
		}
		else {
			for (i32 idx = 0; idx < numInternalNodes; idx++)
			{
				build_lbvh_tree<u32>(numInstances, mortonCodes, parents, idx);
			}
		}
	}


	// Calcualte bounding boxes and SAH.
	// Create atomic counters buffer.
	auto deviceCountersMem = make_udevptr_array<dev, i32>(numInternalNodes);
	i32* deviceCounters = deviceCountersMem.get();
	mem_set<dev>(deviceCounters, 0xFF, numInternalNodes * sizeof(i32));
	// Allocate bounding boxes.
	auto boundingBoxesMem = make_udevptr_array<dev, ei::Vec4>(numNodes * 2);
	ei::Vec4 *boundingBoxes = boundingBoxesMem.get();
	// Allocate collapseOffsets.
	// The last position for collapseOffsets is to avoid access violations,
	// since if the last internal node needs to be collapsed, it will write 
	// to this positions with offset = 0, but this info will not be used further.
	auto collapseOffsetsMem = make_udevptr_array<dev, i32>(numInternalNodes);
	i32* collapseOffsets = collapseOffsetsMem.get();
	mem_set<dev>(collapseOffsets, 0, numInternalNodes * sizeof(i32));
	i32* leafMarks = (i32*)codesMem.get();

	if (dev == Device::CUDA) {
		// Calculate BVH bounding boxes.
		i32 bboxCacheSize = numThreads * sizeof(ei::Vec4) * 2;
		cudaFuncSetCacheConfig(calculate_bounding_boxes_insD, cudaFuncCachePreferShared);
		BoundingBoxFunctor functor;
		get_maximum_occupancy_variable_smem(numBlocks, numThreads, numInstances,
			calculate_bounding_boxes_insD, functor);
		calculate_bounding_boxes_insD << <numBlocks, numThreads, bboxCacheSize >> > (
			numInstances,
			matrices,
			objIds,
			aabbs,
			boundingBoxes,
			primIds,
			parents,
			collapseOffsets,
			traverseCosts.x, traverseCosts.y,
			deviceCounters);

		// Mark all children of collapsed nodes as removed and themselves as leaves (=1).
		get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, mark_nodesD);
		mark_nodesD << <numBlocks, numThreads, bboxCacheSize >> > (
			numInternalNodes,
			boundingBoxes,
			deviceCounters,
			collapseOffsets,
			leafMarks);
	}
	else {
		// Calculate BVH bounding boxes.
		for (i32 idx = 0; idx < numInstances; idx++)
		{
			calculate_bounding_boxes_ins(numInstances,
				matrices,
				objIds,
				aabbs,
				boundingBoxes,
				primIds,
				parents,
				collapseOffsets,
				traverseCosts.x, traverseCosts.y,
				deviceCounters,
				idx);
		}

		// Mark all children of collapsed nodes as removed and themselves as leaves (=1).
		for (i32 idx = 0; idx < numInternalNodes; idx++)
		{
			mark_nodes(numInternalNodes, boundingBoxes, deviceCounters, collapseOffsets, leafMarks, idx);
		}
	}

	i32 numRemovedInternalNodes;
	if (dev == Device::CUDA) {
		// Scan to get values for offsets.
		CuLib::DeviceInclusiveSum(numInternalNodes, &collapseOffsets, &collapseOffsets);
		cudaMemcpy(&numRemovedInternalNodes, collapseOffsets + numInternalNodes - 1, sizeof(i32),
			cudaMemcpyDeviceToHost);
		// Scan to get number of leaves arised from internal nodes before current node.
		CuLib::DeviceExclusiveSum(numInternalNodes + 1, &leafMarks, &leafMarks);
	}
	else {
		// Scan to get values for offsets.
		thrust::inclusive_scan(collapseOffsets, collapseOffsets + numInternalNodes,
			collapseOffsets);
		numRemovedInternalNodes = collapseOffsets[numInternalNodes - 1];
		// Scan to get number of leaves arised from internal nodes before current node.
		thrust::exclusive_scan(leafMarks, leafMarks + numInternalNodes + 1, leafMarks);
	}

	// Here uses a compact memory layout so that each leaf only uses 16 bytes.
	i32 numInternLeavesCollapsedBVH; //!= (numNodesCollapsedBVH + 1) >> 1; since there are simple leaves.
	// and it includes the removed ones.

	if (dev == Device::CUDA)
		cudaMemcpy(&numInternLeavesCollapsedBVH, leafMarks + numInternalNodes, sizeof(i32),
			cudaMemcpyDefault);
	else
		numInternLeavesCollapsedBVH = leafMarks[numInternalNodes];

	i32 numInternalNodesCollapsedBVH = numInternalNodes - numInternLeavesCollapsedBVH;
	numInternLeavesCollapsedBVH -= numRemovedInternalNodes;
	i32 numFloat4InCollapsedBVH = numInternLeavesCollapsedBVH + 4 * numInternalNodesCollapsedBVH;
	//printf("bvhSize %d %d %d\n", bvhSize, 
	//	numInternLeavesCollapsedBVH + numRemovedInternalNodes, numRemovedInternalNodes);
	m_bvhNodes.resize(numFloat4InCollapsedBVH * sizeof(ei::Vec4));
	ei::Vec4* collapsedBVH = as<ei::Vec4>(m_bvhNodes.acquire<dev>());
	// Copy values for collapsed BVH.	
	if (dev == Device::CUDA) {
		get_maximum_occupancy(numBlocks, numThreads, numNodes, copy_to_collapsed_bvh_insD);
		copy_to_collapsed_bvh_insD << < numBlocks, numThreads >> > (
			numNodes,
			numInternalNodes,
			numFloat4InCollapsedBVH - numInternalNodes,
			collapsedBVH,
			boundingBoxes,
			parents,
			deviceCounters,
			primIds,
			leafMarks,
			collapseOffsets
			);
	}
	else {
		for (i32 idx = 1; idx < (i32)numNodes; idx++)
		{
			copy_to_collapsed_bvh_ins(numNodes, numInternalNodes, numFloat4InCollapsedBVH - numInternalNodes, 
				collapsedBVH, boundingBoxes, parents, deviceCounters, primIds, 
				leafMarks, collapseOffsets, idx);
		}
	}
}


template < Device dev >
void LBVHBuilder::build(ObjectDescriptor<dev>& obj, const ei::Box& aabb) {
	ei::Vec4 traverseCosts = { 1.0f, 1.2f, 2.4f, 1.f };
	i32 offsetQuads = obj.polygon.numTriangles;
	i32 offsetSpheres = offsetQuads + obj.polygon.numQuads;
	i32 numPrimitives = offsetSpheres + obj.spheres.numSpheres;

	build_lbvh64<dev>((ei::Vec3*)obj.polygon.vertices, 
		(ei::Vec4*)obj.spheres.spheres, (i32*)obj.polygon.vertexIndices, 
		(i32*)(obj.polygon.vertexIndices + obj.polygon.numTriangles), 
		aabb, traverseCosts,
		numPrimitives, offsetQuads, offsetSpheres);

	m_primIds.mark_changed(dev);
	m_bvhNodes.mark_changed(dev);
}

template void LBVHBuilder::build<Device::CPU>(ObjectDescriptor<Device::CPU>&, const ei::Box&);
template void LBVHBuilder::build<Device::CUDA>(ObjectDescriptor<Device::CUDA>&, const ei::Box&);

template < Device dev >
void LBVHBuilder::build(
	const SceneDescriptor<dev>& scene
) {
	ei::Vec2 traverseCosts = { 1.f, 20.f };// TODO: find value for this.

	build_lbvh32<dev>((ei::Mat3x4*)scene.transformations,
		(i32*)scene.objectIndices,
		(ei::Box*)scene.aabbs,
		scene.aabb,
		traverseCosts, scene.numInstances);
	
	m_primIds.mark_changed(dev);
	m_bvhNodes.mark_changed(dev);
}

template void LBVHBuilder::build<Device::CPU>(const SceneDescriptor<Device::CPU>&);
template void LBVHBuilder::build<Device::CUDA>(const SceneDescriptor<Device::CUDA>&);

}}} // namespace mufflon
