#include "build_lbvh.hpp"
#include "util/types.hpp"
#include "core/cuda/cu_lib_wrapper.h"
#include "core/math/sfcurves.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <device_launch_parameters.h>


namespace mufflon { namespace {//TODO snippers

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

__global__ void calculate_morton_codes64D(
	ei::Vec3* triVertices,
	ei::Vec3* quadVertices,
	ei::Vec4* sphVertices,
	i32* triIndices,
	i32* quadIndices,
	u64* mortonCodes,
	i32* sortIndices, ei::Vec3 lo, ei::Vec3 hi,
	i32 offsetQuads, i32 offsetSpheres, i32 numPrimitives) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numPrimitives)
		return;

	u64 mortonCode;
	ei::Vec3 centroid;
	if (idx >= offsetSpheres) {
		// Calculate Morton codes for spheres.
		i32 sphId = idx - offsetSpheres;
		ei::Vec4 sph = sphVertices[sphId];
		centroid = ei::Vec3(sph);
	}
	else {
		if (idx >= offsetQuads) {
			// Calculate Morton codes for quads.
			i32 quadId = (idx - offsetQuads) << 2;
			ei::Vec3 v0 = quadVertices[quadIndices[quadId]];
			ei::Vec3 v1 = quadVertices[quadIndices[quadId + 1]];
			ei::Vec3 v2 = quadVertices[quadIndices[quadId + 2]];
			ei::Vec3 v3 = quadVertices[quadIndices[quadId + 3]];
			centroid = get_quad_centroid(v0, v1, v2, v3);
		}
		else {
			// Calculate Morton codes for triangles.
			i32 triId = idx * 3;
			ei::Vec3 v0 = triVertices[triIndices[triId]];
			ei::Vec3 v1 = triVertices[triIndices[triId + 1]];
			ei::Vec3 v2 = triVertices[triIndices[triId + 2]];
			centroid = get_triangle_centroid(v0, v1, v2);
		}
	}
	ei::Vec3 normalizedPos = normalize_position(centroid, lo, hi);
	mortonCode = calculate_morton_code64(normalizedPos);
	mortonCodes[idx] = mortonCode;
	sortIndices[idx] = idx;
}

template<typename T> 
void get_maximum_occupancy(i32 &gridSize, i32 &blockSize, i32 totalThreads, T func, 
						   i32 dynamicSMemSize = 0)
{
	//blockSize;   // The launch configurator returned block size 
	i32 minGridSize; // The minimum grid size needed to achieve the 
	// maximum occupancy for a full device launch 
	//gridSize;    // The actual grid size needed, based on input size 

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, 0);

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

	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, func, blockSizeToDynamicSMemSize, 0);

	if (blockSize != 0)
		// Round up according to array size 
		gridSize = (totalThreads + blockSize - 1) / blockSize;
	else
		gridSize = minGridSize;
}

__device__ i32 longestCommonPrefix(u64* sortedKeys,
	u32 numberOfElements, i32 index1, i32 index2, u64 key1)
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
		return 64 + __clzll(index1 ^ index2);
	}

	return __clzll(key1 ^ key2);
}

__device__ i32 sgn(i32 number)
{
	return (0 < number) - (0 > number);
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
	const i32 maxDivisor = 1 << (32 - __clz(l));
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
		rightIndex = splitPosition;
	}

	// Set parent nodes.
	parents[leftIndex] = -idx;
	parents[rightIndex] = idx;

	// Set the parent of the root node to -1.
	if (idx == 0)
	{
		parents[0] = 0xEFFFFFFF;
	}
}


__device__ __forceinline__
void extract_prim_counts(i32 primitiveCount, ei::IVec4& count) {
	i32 sphCountMask = 0x000003FF;
	i32 triCountMask = 0x3FF00000;
	i32 quadCountMask = 0x000FFC00;
	i32 triShift = 20;
	i32 quadShift = 10;
	count.x = (primitiveCount & triCountMask) >> triShift;
	count.y = (primitiveCount & quadCountMask) >> quadShift;
	count.z = (primitiveCount & sphCountMask);
	count.w = count.x + count.y + count.z;
}

struct BBCache {
	ei::Box bb;
	float cost;
	i32 primCount;
};
static_assert(sizeof(BBCache) == 8*sizeof(float), "Alignment of BBCache will be broken.");

__global__ void calculate_bounding_boxesD(
	u32 numPrimitives,
	ei::Vec3* triVertices,
	ei::Vec3* quadVertices,
	ei::Vec4* sphVertices,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec4 * __restrict__ boundingBoxes, //TODO remove __restricts?
	i32 *sortedIndices,
	i32 * __restrict__ parents,
	i32 *collapseOffsets,
	i32 offsetQuads, i32 offsetSpheres,
	float ci, float ct0, float ct1, float ct2,
	u32* counters)
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

	// If node is to be collapsed, offsets, dataIndices need to be updated.
	bool checkCollapse = true;
	i32 primId = sortedIndices[idx];

	// Calculate leaves bounding box and set primitives count and intersection test cost.
	ei::Box currentBb;
	float cost, costAsLeaf;
	// primitiveCount stores the numbers of each primitives in form:
	// 2 unused bits + 10 bits triangle + 10 bits quads + 10 bits spheres.
	i32 primitiveCount;
	if (primId >= offsetSpheres) {
		// Calculate bounding box for spheres.
		i32 sphId = primId - offsetSpheres;
		ei::Sphere sph = *reinterpret_cast<ei::Sphere*>(sphVertices+sphId);
		currentBb = ei::Box(sph);
		primitiveCount = 1;
		cost = ct2;
	}
	else {
		if (primId >= offsetQuads) {
			// Calculate bounding box for quads.
			i32 quadId = (primId - offsetQuads) << 2;
			ei::Vec3 v[4] = { quadVertices[quadIndices[quadId]],
							  quadVertices[quadIndices[quadId + 1]],
							  quadVertices[quadIndices[quadId + 2]],
							  quadVertices[quadIndices[quadId + 3]] };
			currentBb = ei::Box(v, 4);
			primitiveCount = 0x00000400;
			cost = ct1;
		}
		else {
			// Calculate bounding box for triangles.
			i32 triId = primId * 3;
			ei::Vec3 v[3] = { triVertices[triIndices[triId]],
							  triVertices[triIndices[triId + 1]],
							  triVertices[triIndices[triId + 2]] };
			currentBb = ei::Box(v, 3);
			primitiveCount = 0x00100000;
			cost = ct0;
		}
	}

	// Store cost and primitiveCount.
	cost *= ei::surface(currentBb);

	// Update node bounding boxes of current node.
	sharedBb[threadIdx.x] = {currentBb, cost, primitiveCount};
	i32 leafIndex = idx + (numPrimitives - 1); 
	i32 boxId = leafIndex << 1;
	boundingBoxes[boxId] = {currentBb.min, __int_as_float(primitiveCount)};
	boundingBoxes[boxId + 1] = {currentBb.max, cost};

	__syncthreads(); // TODO check this.

	// Initialize.
	i32 current = parents[leafIndex];
	bool lastNodeIsLeftChild = false;
	if (current < 0) {
		current = -current;
		lastNodeIsLeftChild = true;
	}
	i32 lastNode; // Does not need to be initialized, since leaves will not be collapsed
				  // due to positive values of primitiveCount
	// Some auxilary variables for calculating primitiveCount.
	ei::IVec4 counts; // x: tri; y: quad; z: sphere; w: total.
	ei::IVec4 otherCounts; // x: tri; y: quad; z: sphere; w: total.

	while (true)
	{
		// In the counters array, we have stored the id of the thread that processed the other
		// children of this node.
		u32 childThreadId = atomicExch(&counters[current], idx);

		// The first thread to reach a node will just die.
		if (childThreadId == 0xFFFFFFFF)
		{
			return;
		}

		// Fetch bounding boxes and counts information.
		BBCache childInfo;
		if (childThreadId >= firstThreadInBlock && childThreadId <= lastThreadInBlock) {
			// If both child nodes were processed by the same block, we can reuse the values
			// cached in shared memory.
			i32 childThreadIdInBlock = childThreadId - firstThreadInBlock;
			childInfo = sharedBb[childThreadIdInBlock];
		} else {
			// The children were processed in different blocks, so we have to find out if the one
			// that was not processed by this thread was the left or right one.
			boxId = childThreadId << 1;
			ei::Vec4 childBbMin = boundingBoxes[boxId];
			ei::Vec4 childBbMax = boundingBoxes[boxId + 1];
			childInfo = BBCache{
				ei::Box{ei::Vec3{childBbMin}, ei::Vec3{childBbMax}},
				childBbMax.w, __float_as_int(childBbMin.w)
			};
		}

		__syncthreads(); // @todo check.

		if (checkCollapse) {
			// Calculate primitves counts.
			// Set offsets.
			if (primitiveCount < 0) { // Count < 0 means the node should be collapsed.
				primitiveCount = -primitiveCount;
				// offset is the number of internal nodes below the child lastNode.
				i32 offset = lastNode - counts.w + 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[lastNode] = offset;
				else
					// Since lastNode as right child should be collapsed, lastNode + 1
					// must be one child of lastNode if it has more than 2 primitves.
					collapseOffsets[lastNode + 1] = offset;
			}

			extract_prim_counts(childInfo.primCount, otherCounts);

			if (childInfo.primCount < 0) {
				childInfo.primCount = -childInfo.primCount;
				// offset is the number of internal nodes below the other child.
				i32 offset = childThreadId - otherCounts.w + 2;
				if (lastNodeIsLeftChild)
					collapseOffsets[childThreadId + 1] = offset;
				else
					// Since theOtherChild as right child should be collapsed, lastNode + 1
					// must be one child of theOtherChild if it has more than 2 primitves.
					collapseOffsets[childThreadId] = offset;
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
				primitiveCount = __int_as_float(0x00000FFF);
			} else {
				primitiveCount += childInfo.primCount;
				// Calculate costs.
				float area = ei::surface(currentBb);
				cost = ci * area + childInfo.cost + cost;
				counts.w = counts.x + counts.y + counts.z;// Determine offset.
				costAsLeaf = area * (counts.x * ct0 + counts.y * ct1 + counts.z * ct2);
				if (costAsLeaf < cost) {
					// Collapse.
					primitiveCount = -primitiveCount;
					// Update cost.
					cost = costAsLeaf;
				}
			}
		}

		// Update last processed node
		lastNode = current;

		// Update current node pointer
		current = parents[current];
		if (current < 0) {
			current = -current;
			lastNodeIsLeftChild = true;
		} else {
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
		currentBb = ei::Box{ currentBb, childInfo.bb };
		sharedBb[threadIdx.x] = BBCache{currentBb, cost, primitiveCount};
		boxId = lastNode << 1;
		boundingBoxes[boxId] = {currentBb.min, __int_as_float(primitiveCount)};
		boundingBoxes[boxId + 1] = {currentBb.max, cost};

		__syncthreads(); //@todo check.

		if (current == 0xEFFFFFFF) {
			// Print the bounding box of the base node.
			printf("root bounding box:\n%f %f %f\n%f %f %f\n",
				currentBb.min.x, currentBb.min.y, currentBb.min.z,
				currentBb.max.x, currentBb.max.y, currentBb.max.z);
			return;
		}
	}
}

__global__ void mark_nodesD(
	u32 numInternalNodes,
	ei::Vec4* __restrict__ boundingBoxes,
	i32* __restrict__ collapsedOffsets,
	i32* __restrict__ removedMarks,
	i32* __restrict__ leafMarks
) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numInternalNodes)
		return;

	// This is better than using a stack, since less operations are needed
	// and due to parallerism, performance is not affected.
	i32 boxId = idx << 1;
	i32 primitiveCount = __float_as_int(boundingBoxes[boxId].w);
	if (primitiveCount >= 0) {
		leafMarks[idx] = 0;
		return;
	}
	else {
		leafMarks[idx] = 1;
	}
	ei::IVec4 count;
	// Extract counts for three kinds of primitvies.
	extract_prim_counts(primitiveCount, count);
	//if (count.w == 2) // Not needed due to start <= end check.
	//	return; 
	i32 start, end;
	if (primitiveCount & 0x40000000 == 0) {
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
		++start;
	}
}


__device__ __forceinline__
i32 insert_id(i32 id, i32* preLeaves) {
	i32 numPreLeaves = preLeaves[id];
	return (id << 2) - numPreLeaves * 3;
}

__device__ __forceinline__
i32 next_id(ei::IVec4& insertPos, ei::IVec2& endPos, i32& primType) {
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

__global__ void copy_to_collapsed_bvh(
	i32 numInternalNodes,
	ei::Vec4* collapsedBVH,
	ei::Vec4 * __restrict__ boundingBoxes,
	i32* __restrict__ parents,
	i32* __restrict__ removedMarks,
	i32* __restrict__ sortIndices,
	i32* __restrict__ preLeaves,
	i32* __restrict__ reduceOffsets,
	i32 offsetQuads, i32 offsetSpheres
) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numInternalNodes)
		return;

	bool isLeftChilld = false;
	if (removedMarks[idx] != 0xFFFFFFFF) {
		i32 parent = parents[idx];
		i32 offset = 1;
		if (parent < 0) {
			isLeftChilld = true;
			parent = -parent;
			offset = 0;
		}

		// Copy bounding boxes to parent.
		// Set to parent id in collapsed bvh.
		i32 id = parent - reduceOffsets[parent]; 
		i32 nodeId = idx << 1;
		ei::Vec4 lo = boundingBoxes[nodeId];
		ei::Vec4 hi = boundingBoxes[nodeId + 1];
		i32 insertId = insert_id(id, preLeaves);
		collapsedBVH[insertId + offset] = ei::Vec4(lo.x, hi.x, lo.y, hi.y);
		*(((ei::Vec2*)&collapsedBVH[insertId + 2]) + offset) = ei::Vec2(lo.z, hi.z);

		// Set child pointers to parent.
		// set to current node id in collapsed bvh.
		id = idx - reduceOffsets[idx]; 
		insertId = insert_id(id, preLeaves);
		if (lo.w < 0) { // Leaf with negative sign to the pointer.
			insertId = ~insertId;
		}
		*(((i32*)&collapsedBVH[nodeId + 3]) + offset) = insertId;
		
		// Copy data indices.
		if (lo.w < 0) {
			i32 primitiveCount = lo.w;
			ei::IVec4 counts;
			extract_prim_counts(primitiveCount, counts);
			i32 startId;
			if (isLeftChilld) {
				startId = idx - counts.w + 1;
			}
			else {
				startId = idx;
			}

			// Now counts is used to register insert position.
			ei::IVec2 endPos;
			endPos.x = startId + counts.x;
			endPos.y = endPos.x + counts.y;
			i32 primId;
			i32 nextId;
			i32 insertId;
			bool readId = true;
			counts.x = startId;
			counts.y = endPos.x;
			counts.z = endPos.y;
			//collapsedBVH[id] = ei::Vec4(primitiveCount, counts);// Error.
			collapsedBVH[id] = ei::Vec4(primitiveCount, counts.x, counts.y, counts.z);
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
						if (primType == 0) {
							goto stay_at_place;
						}
						primType = 2;
					}
					readId = false;
				}

				//Now primType matches current primId.
				insertId = counts[primType]++;

				tmpId = sortIndices[insertId];
				--counts.w;
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
				else if (primType == 2){
					continue;
				}
				else {
					primType = 2;
				}

				sortIndices[insertId] = primId;
				primId = tmpId;
				continue;
			stay_at_place:
				--counts.w; // Reduce the numbe of to be inserted primitives.
				counts[primType]++;
			}
		}
	}
}

}} // namespace mufflon:: {



namespace mufflon { namespace scene { namespace accel_struct {

ei::Vec4* build_lbvh64(ei::Vec3* triVertices,
	ei::Vec3* quadVertices,
	ei::Vec4* sphVertices,
	i32* triIndices,
	i32* quadIndices,
	ei::Vec3 lo, ei::Vec3 hi, ei::Vec4 traverseCosts,
	i32 numTriangles, i32 numQuads, i32 numSpheres) {
	// Calculate offsets for each kind of primitives.
	i32 offsetQuads = numTriangles;
	i32 offsetSpheres = offsetQuads + numQuads;
	i32 numPrimitives = offsetSpheres + numSpheres;

	i32 numBlocks, numThreads;

	// Calculate Morton codes.
	u64* mortonCodes;
	cudaMalloc((void**)&mortonCodes, numPrimitives * sizeof(u64));
	i32* sortIndices;
	cudaMalloc((void**)&sortIndices, numPrimitives * sizeof(i32));
	get_maximum_occupancy(numBlocks, numThreads, numPrimitives, calculate_morton_codes64D);
	calculate_morton_codes64D <<< numBlocks, numThreads >>> (triVertices, quadVertices, 
		sphVertices, triIndices, quadIndices, 
		mortonCodes, sortIndices, lo, hi, 
		offsetQuads, offsetSpheres, numPrimitives);

	// Sort based on Morton codes.
	CuLib::DeviceSort(numPrimitives, &mortonCodes, &mortonCodes,
		&sortIndices, &sortIndices);

	// Create BVH.
	// Layout: first internal nodes, then leves.
	i32 numInternalNodes = numPrimitives - 1;
	u32 numNodes = numInternalNodes + numPrimitives;
	i32 *parents; // size numNodes.
	u32 totalBytes = numNodes * sizeof(i32);
	cudaMalloc((void**)&parents, totalBytes);
	cudaFuncSetCacheConfig(build_lbvh_treeD<u64>, cudaFuncCachePreferL1); 
	get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, build_lbvh_treeD<u64>);
	build_lbvh_treeD<<<numBlocks, numThreads >>>(//TODO check <u64>
		numPrimitives,
		mortonCodes,
		parents);

	// Calcualte bounding boxes and SAH.
	// Create atomic counters buffer.
	u32* deviceCounters;
	cudaMalloc((void**)&deviceCounters, (numPrimitives - 1) * sizeof(u32));
	cudaMemset(deviceCounters, 0xFF, (numPrimitives - 1) * sizeof(u32));
	// Allocate bounding boxes.
	ei::Vec4 *boundingBoxes;
	cudaMalloc((void**)&boundingBoxes, numNodes * 2 * sizeof(ei::Vec4));
	// Allocate collapseOffsets.
	i32* collapseOffsets;
	cudaMalloc((void**)& collapseOffsets, (numPrimitives - 1) * sizeof(i32));
	// Calculate BVH bounding boxes.
	i32 bboxCacheSize = numThreads * sizeof(ei::Vec4) * 2;
	cudaFuncSetCacheConfig(calculate_bounding_boxesD, cudaFuncCachePreferShared);
	BoundingBoxFunctor functor;
	get_maximum_occupancy_variable_smem(numBlocks, numThreads, numPrimitives,
		calculate_bounding_boxesD, functor);
	calculate_bounding_boxesD <<<numBlocks, numThreads, bboxCacheSize >>> (
		numPrimitives,
		triVertices, quadVertices,
		sphVertices, triIndices, quadIndices,
		boundingBoxes,
		sortIndices,
		parents,
		collapseOffsets,
		offsetQuads, offsetSpheres,
		traverseCosts.x, traverseCosts.y, traverseCosts.z, traverseCosts.w,
		deviceCounters);

	// Mark all children of collapsed nodes as removed and themselves as leaves (=1).
	get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, mark_nodesD);
	i32* removedMarks = (i32*)deviceCounters;
	i32* leafMarks = (i32*)mortonCodes; 
	mark_nodesD << <numBlocks, numThreads, bboxCacheSize >> > (
		numInternalNodes,
		boundingBoxes,
		collapseOffsets,
		removedMarks,
		leafMarks);

	// Scan to get values for offsets.
	CuLib::DeviceInclusiveSum(numInternalNodes, &collapseOffsets, &collapseOffsets);
	i32 numNodesCollapsedBVH;
	cudaMemcpy(&numNodesCollapsedBVH, collapseOffsets + numInternalNodes - 1, sizeof(i32),
		cudaMemcpyDeviceToHost);
	numNodesCollapsedBVH = numInternalNodes - numNodesCollapsedBVH;

	// Scan to get number of leaves before current node.
	CuLib::DeviceExclusiveSum(numInternalNodes, &leafMarks, &leafMarks);

	// Copy values for collapsed BVH.	
	ei::Vec4* collapsedBVH;
	// Here uses a compact memory layout so that each leaf only uses 16 bytes.
	// Todo check value of numLeavesCollapsedBVH using the last + 1 element of leafMarks
	i32 numLeavesCollapsedBVH = (numNodesCollapsedBVH + 1) >> 1;
	i32 numInternalNodesCollapsedBVH = numNodesCollapsedBVH - numLeavesCollapsedBVH;
	cudaMalloc((void**)&collapsedBVH, (numLeavesCollapsedBVH + 4 * numInternalNodesCollapsedBVH) 
		* sizeof(ei::Vec4));
	get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, copy_to_collapsed_bvh);
	copy_to_collapsed_bvh << < numBlocks, numThreads >> > (
		numInternalNodes,
		collapsedBVH,
		boundingBoxes,
		parents,
		removedMarks,
		sortIndices,
		leafMarks,
		collapseOffsets,
		offsetQuads, offsetSpheres
		);


	// Free device memory.
	cudaFree(leafMarks);// aka mortonCodes.
	cudaFree(sortIndices);
	cudaFree(parents);
	cudaFree(removedMarks);// aka deviceCounters.
	cudaFree(boundingBoxes);
	cudaFree(collapseOffsets);

	return collapsedBVH;
}
}}} // namespace mufflon
