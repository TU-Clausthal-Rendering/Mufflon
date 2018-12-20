#include "lbvh.hpp"
#include "util/types.hpp"
#include "core/cuda/cu_lib_wrapper.hpp"
#include "core/cuda/cuda_utils.hpp"
#include "core/math/sfcurves.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/allocator.hpp"
#include "accel_structs_commons.hpp"

#include <cuda_runtime_api.h>
#include <ei/3dtypes.hpp>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>




namespace mufflon {
namespace scene {
namespace accel_struct {


static_assert(MAX_ACCEL_STRUCT_PARAMETER_SIZE >= sizeof(LBVH),
	"Descriptor parameter block to small for this acceleration structure.");



// Type trait to derive some dependent types from a descriptor.
template<typename Desc> struct desc_info {};
template<Device dev> struct desc_info<ObjectDescriptor<dev>> {
	using MortonCode = u64;
	using PrimCount = ei::IVec3;
	using CostFactor = ei::Vec4;
	static constexpr float NODE_TRAVERSAL_COST = 1.0f;
	static constexpr ei::Vec3 PRIM_TRAVERSAL_COST = { 1.2f, 2.4f, 1.0f };
};
template<Device dev> struct desc_info<SceneDescriptor<dev>> {
	using MortonCode = u32;
	using PrimCount = ei::Vec<i32, 1>;
	using CostFactor = ei::Vec2;
	static constexpr float NODE_TRAVERSAL_COST = 1.0f;
	static constexpr ei::Vec<float, 1> PRIM_TRAVERSAL_COST { 20.0f };// TODO: find value for this.
};
template<typename Desc>
using MortonCode_t = typename desc_info<Desc>::MortonCode;
template<typename Desc>
using PrimCount_t = typename desc_info<Desc>::PrimCount;
template<typename Desc>
using CostFactor_t = typename desc_info<Desc>::CostFactor;


// TODO: move to math header ----
// Calculates the point morton code using 63 bits.
template<typename Code>
__forceinline__ __host__ __device__ Code calculate_morton_code(const ei::Vec3& point) {}
template<>
__forceinline__ __host__ __device__ u32 calculate_morton_code<u32>(const ei::Vec3& point) {
	// Discretize the unit cube into a 10 bit integer
	ei::UVec3 discretized{ ei::clamp(point * 1024.0f, 0.0f, 1023.0f) };

	return math::part_by_two10(discretized[0]) * 4
		+ math::part_by_two10(discretized[1]) * 2
		+ math::part_by_two10(discretized[2]);
}
template<>
__forceinline__ __host__ __device__ u64 calculate_morton_code<u64>(const ei::Vec3& point) {
	// Discretize the unit cube into a 21 bit integer
	ei::UVec3 discretized { ei::clamp(point * 2097152.0f, 0.0f, 2097151.0f) };

	return math::part_by_two21(discretized[0]) * 4
		 + math::part_by_two21(discretized[1]) * 2
		 + math::part_by_two21(discretized[2]);
}
// ----

__forceinline__ __host__ __device__
ei::Vec3 normalize_position(ei::Vec3 pos, const ei::Box& box) {
	ei::Vec3 span = box.max - box.min;
	return (pos - box.min) / span;
}

template<typename DescType>
CUDA_FUNCTION MortonCode_t<DescType>
calculate_morton_code(const DescType& primitives, i32 idx,
					  const ei::Box& sceneBB) {
	const ei::Vec3 centroid = get_centroid(primitives, idx);
	const ei::Vec3 normalizedPos = normalize_position(centroid, sceneBB);
	return calculate_morton_code<MortonCode_t<DescType>>(normalizedPos);
}


template < typename DescType >
__global__ void calculate_morton_codesD(
	const DescType& desc,
	const ei::Box& sceneBB,
	const i32 numPrimitives,
	MortonCode_t<DescType>* mortonCodes,
	i32* sortIndices) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= numPrimitives)
		return;

	mortonCodes[idx] = calculate_morton_code(desc, idx, sceneBB);
	sortIndices[idx] = idx;
}


// TODO --- cleanup and move to some CUDA header
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
// -----

template<typename Key>
CUDA_FUNCTION i32 longestCommonPrefix(Key* sortedKeys,
	i32 numberOfElements, i32 index1, i32 index2, Key key1)
{
	// No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
	// thread per internal node)
	if (index2 < 0 || index2 >= numberOfElements)
		return 0;

	Key key2 = sortedKeys[index2];

	if (key1 == key2)
		return 64 + (i32)cuda::clz(u32(index1 ^ index2));

	return (i32)cuda::clz(key1 ^ key2);
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

	const i32 direction = ei::sgn(lcp1 - lcp2);

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
	const i32 maxDivisor = 1 << (32 - cuda::clz(u32(l)));
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
		return;

	build_lbvh_tree<T>(numPrimitives, sortedKeys, parents, idx);
}

struct BBCache {
	BBCache() = default;
	CUDA_FUNCTION __forceinline__ BBCache(const ei::Vec4& a, const ei::Vec4& b) :
		boxMin{a}, cost{a.w},
		boxMax{b}, primCount{float_bits_as_int(b.w)}
	{}
	CUDA_FUNCTION __forceinline__ BBCache(const ei::Vec3& boxMin,
							float cost,
							const ei::Vec3& boxMax,
							i32 primCount) :
		boxMin(boxMin), cost(cost), boxMax(boxMax), primCount(primCount)
	{}
	ei::Vec3 boxMin;
	float cost;
	ei::Vec3 boxMax;
	i32 primCount;
};
static_assert(sizeof(BBCache) == 8*sizeof(float), "Alignment of BBCache is broken.");


template < typename DescType > CUDA_FUNCTION
PrimCount_t<DescType> get_count(const DescType&, i32 primIdx) { return PrimCount_t<DescType>{1}; }
template < Device dev > CUDA_FUNCTION
PrimCount_t<ObjectDescriptor<dev>> get_count(const ObjectDescriptor<dev>& obj, i32 primIdx) {
	if(primIdx >= i32(obj.polygon.numTriangles + obj.polygon.numQuads))
		return {0, 0, 1};
	if(primIdx >=  i32(obj.polygon.numTriangles))
		return {0, 1, 0};
	return {1, 0, 0};
}

template< typename PrimCount > CUDA_FUNCTION
i32 encode_prim_counts(const PrimCount& primCount) {
	return primCount.x;
}
template<> CUDA_FUNCTION
i32 encode_prim_counts(const ei::IVec3& primCount) {
	return (ei::min(primCount.x, 0x3FF) << 20)
		 | (ei::min(primCount.y, 0x3FF) << 10)
		 | (ei::min(primCount.z, 0x3FF));
}

template< typename PrimCount > CUDA_FUNCTION
PrimCount extract_prim_counts(i32 primCount) {
	return PrimCount{primCount};
}
template<> CUDA_FUNCTION
ei::IVec3 extract_prim_counts(i32 primCount) {
	return ei::IVec3{(primCount & 0x3FF00000) >> 20,
					 (primCount & 0x000FFC00) >> 10,
					 (primCount & 0x000003FF)};
}

template < typename DescType >
CUDA_FUNCTION float get_cost(const PrimCount_t<DescType>& primCount) {
	const auto traversalCost = desc_info<DescType>::PRIM_TRAVERSAL_COST;
	return desc_info<DescType>::NODE_TRAVERSAL_COST
		+ dot(primCount, traversalCost);
}

template < typename DescType >
CUDA_FUNCTION void calculate_bounding_boxes(
	const DescType& desc,
	i32 idx,		// Global thread index
	i32 primIdx,	// Index of the primitive for which this thread is responsible
	i32 numPrimitives,
	const i32* __restrict__ parents,
	ei::Vec4* boundingBoxes,
	i32* __restrict__ counters,
	i32* __restrict__ nodeMarks,
	const i32 firstThreadInBlock = 0,
	const i32 lastThreadInBlock = 0,
	BBCache* sharedBb = nullptr
) {
	ei::Box currentBb = get_bounding_box(desc, primIdx);

	// Get primitive count of the node and its cost
	auto primitiveCount = get_count(desc, primIdx);
	float cost = get_cost<DescType>(primitiveCount) * ei::surface(currentBb);

	// Store in cache and global array
#ifdef __CUDA_ARCH__
	sharedBb[threadIdx.x] = { currentBb.min, cost, currentBb.max, encode_prim_counts(primitiveCount) };
#endif
	const i32 numInternalNodes = numPrimitives - 1;
	const i32 leafIndex = idx + numInternalNodes;
	const i32 boxId = leafIndex << 1;
	boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(encode_prim_counts(primitiveCount)) };
	boundingBoxes[boxId + 1] = { currentBb.max, cost };

	cuda::syncthreads(); // For shared memory cache

	// Proceed upwards in the hierarchy
	i32 currentNode = parents[leafIndex];
	i32 lastNode = idx; // Only for the first iteration is is the thread index, later it is realy a node
	bool lastIsLeftChild = currentNode < 0;
	if(currentNode < 0) currentNode = ~currentNode;

	// In the counters array, we have stored the id of the thread that processed the other
	// children of this node.
#ifdef __CUDA_ARCH__
	// Need to write the node index (instead real thread index) to communicate
	// that this was a leaf.
	i32 otherChildThreadIdx = atomicExch(&counters[currentNode], leafIndex);
#else
	i32 otherChildThreadIdx = counters[currentNode]; 
	counters[currentNode] = leafIndex;
#endif // __CUDA_ARCH__

	// The first thread to reach a node will just die.
	// This circumvents the global sync problem. The second thread
	// can be sure that the data of the first one is present.
	while(otherChildThreadIdx != 0xFFFFFFFF) { // TODO: do while?
		cuda::globalMemoryBarrier();		// For reads on boundingBoxes[]

		i32 otherChildNode = lastIsLeftChild ? lastNode + 1 : lastNode - 1;
		// otherChildNode may be invalid if it was a leaf - this is encoded
		// in the otherChildThreadIdx.
		if(otherChildThreadIdx >= numInternalNodes) {
			otherChildThreadIdx -= numInternalNodes;	// Remove the encoding, such that it is an index
			otherChildNode += numInternalNodes;			// Get the correct other node
		}

		BBCache childInfo;
#ifdef __CUDA_ARCH__
		// Read from cache possible, iff in the same block as the current thread
		if(otherChildThreadIdx >= firstThreadInBlock && otherChildThreadIdx <= lastThreadInBlock)
			childInfo = sharedBb[otherChildThreadIdx - firstThreadInBlock];
		else {
			i32 boxId = otherChildNode * 2;
			childInfo = BBCache{boundingBoxes[boxId], boundingBoxes[boxId+1]};
		}
#else
		// The children were processed in different blocks, so we have to find out if the one
		// that was not processed by this thread was the left or right one.
		{
			i32 boxId = otherChildNode * 2;
			childInfo = BBCache{boundingBoxes[boxId], boundingBoxes[boxId+1]};
		}
#endif

		cuda::syncthreads(); // For cached reads

		// Compute data for this node
		currentBb.min = ei::min(currentBb.min, childInfo.boxMin);
		currentBb.max = ei::min(currentBb.max, childInfo.boxMax);
		primitiveCount += extract_prim_counts<PrimCount_t<DescType>>(childInfo.primCount);
		cost = desc_info<DescType>::NODE_TRAVERSAL_COST * ei::surface(currentBb) + cost + childInfo.cost;
		i32 boxId = currentNode << 1;
		boundingBoxes[boxId] = { currentBb.min, int_bits_as_float(encode_prim_counts(primitiveCount)) };
		boundingBoxes[boxId + 1] = { currentBb.max, cost };

		// Go to next node
		lastNode = currentNode;
		currentNode = parents[currentNode];
		lastIsLeftChild = currentNode < 0;
		if(currentNode < 0) currentNode = ~currentNode;
#ifdef __CUDA_ARCH__
		otherChildThreadIdx = atomicExch(&counters[currentNode], idx);
#else
		otherChildThreadIdx = counters[currentNode]; 
		counters[currentNode] = idx;
#endif // __CUDA_ARCH__
	}

	// Initialize nodeMarks for the next kernel.
	if(idx < numPrimitives-1) // Only for internal nodes
		nodeMarks[idx] = 1;
}

// TODO: epsilon scalar overloads
CUDA_FUNCTION __forceinline__ bool greatereq(i32 a, i32 b) { return a >= b; }
CUDA_FUNCTION __forceinline__ bool any(bool a) { return a; }

template< typename DescType >
CUDA_FUNCTION void mark_collapsed_nodes(
	const ei::Vec4* __restrict__ boundingBoxes,
	const i32* __restrict__ parents,
	const i32 leafIndex,	// Global thread index + numInternalNodes
	i32* nodeMarks			// Initialized to 1 (do not delete this node)
) {
	// 1. Pass: Bottom-Up traversal as long as nodes are canditates for collapses.
	// Goal: find the highest collapsable node (ancestor) of the current leaf.
	i32 collapseNode = -1;
	i32 currentNode = parents[leafIndex];
	if(currentNode < 0) currentNode = ~currentNode;
	while(true) {
		ei::Vec4 boxMin_primCount = boundingBoxes[currentNode * 2];
		ei::Vec4 boxMax_primCost = boundingBoxes[currentNode * 2 + 1];
		ei::Box currentBb {ei::Vec3{boxMin_primCount}, ei::Vec3{boxMax_primCost}};
		auto primitiveCount = extract_prim_counts<PrimCount_t<DescType>>(float_bits_as_int(boxMin_primCount.w));

		// Termination condition: more than 1023 primitves of a single type.
		// This is the largest possible value using our encoding.
		if(any(greatereq(primitiveCount, 1023)))
			break;

		float cost = boxMax_primCost.w;		// Cost without collapse
		float costAsLeaf = ei::surface(currentBb) * get_cost<DescType>(primitiveCount);
		if(costAsLeaf < cost)
			collapseNode = currentNode;

		currentNode = parents[currentNode];
		if(currentNode < 0) currentNode = ~currentNode;
	}

	// Is there any ancestor, which is collapsed?
	if(collapseNode != -1) {
		// Mark the current node and all nodes up to the ancestor as deleted.
		i32 currentNode = parents[leafIndex];
		if(currentNode < 0) currentNode = ~currentNode;
		while(currentNode != collapseNode) {
			nodeMarks[currentNode] = 0;
			currentNode = parents[currentNode];
			if(currentNode < 0) currentNode = ~currentNode;
		}
		nodeMarks[currentNode] = 0; // Include a mark for the collapsed node
	}
}


template< typename DescType >
__global__ void calculate_bounding_boxesD(
	const DescType& desc,
	const i32 numPrimitives,
	const i32* __restrict__ sortedIndices,
	const i32* __restrict__ parents,
	ei::Vec4* boundingBoxes,
	i32* __restrict__ counters,
	i32* __restrict__ nodeMarks
) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	const i32 firstThreadInBlock = blockIdx.x * blockDim.x;
	const i32 lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

	// Initialize cache of bounding boxes in shared memory.
	extern __shared__ BBCache sharedBb[]; 

	// Check for valid threads.
	if (idx >= numPrimitives)
		return;

	calculate_bounding_boxes<DescType>(desc, idx, sortedIndices[idx],
		numPrimitives, parents, boundingBoxes, counters, nodeMarks,
		firstThreadInBlock, lastThreadInBlock, sharedBb);
}

template< typename DescType >
__global__ void mark_nodesD(
	u32 numInternalNodes,
	const ei::Vec4* __restrict__ boundingBoxes,
	const i32* __restrict__ parents,
	i32* __restrict__ nodeMarks
) {
	const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx > numInternalNodes) // idx == numInternalNodes OK, because we want one thread per leaf
		return;

	mark_collapsed_nodes<DescType>(boundingBoxes, parents,
		idx + numInternalNodes, nodeMarks);
}


CUDA_FUNCTION bool is_collapsed(const i32* offsets, const i32 numInternalNodes, const i32 node) {
	bool isCollapsed = false;
	if(node > 0 && node < numInternalNodes) // Root and leaf nodes cannot be collapsed.
		// Revert the inclusive scan to check wether this node had a mark '1' or '0'
		isCollapsed = (offsets[node] - offsets[node-1]) == 0; // 0 means collapse
	return isCollapsed;
}

// Called for all nodes
template < typename DescType >
CUDA_FUNCTION void copy_to_collapsed_bvh(
	const ei::Vec4* __restrict__ boundingBoxes,
	const i32* __restrict__ parents,
	const i32* __restrict__ offsets,
	const i32 node,
	const i32 numInternalNodes,
	ei::Vec4* __restrict__ collapsedBVH
) {
	// Collapsed nodes do not exist anymore and do not write anything in the
	// hierarchy
	if(!is_collapsed(offsets, numInternalNodes, node)) {
		// Read information of the current node and determine if it is a left or right
		// child. A child will fill half of the data of the parent. To determine which
		// half is written we use 'offset'.
		ei::Vec4 boxMin_primCount = boundingBoxes[node * 2];
		ei::Vec4 boxMax_primCost = boundingBoxes[node * 2 + 1];
		i32 countCode = float_bits_as_int(boxMin_primCount.w);
		i32 parent = parents[node];
		i32 offset;
		if(parent < 0) { // Left child?
			offset = 0;
			parent = ~parent;
		} else offset = 1;

		// The parent could be collapsed if the current node is a leaf.
		// Search the first non-collapsed parent.
		if(node >= numInternalNodes) {
			i32 last = node;
			while(is_collapsed(offsets, numInternalNodes, parent)) {
				last = parent;
				parent = parents[parent];
				if(parent < 0) { // Left child?
					offset = 0;
					parent = ~parent;
				} else offset = 1;
			}
			countCode = float_bits_as_int(boundingBoxes[last * 2].w);
		}

		i32 outIdx = offsets[parent] * 4;
		collapsedBVH[outIdx+offset] = ei::Vec4{boxMin_primCount.x,  boxMax_primCost.x,  boxMin_primCount.y,  boxMax_primCost.y};
		((ei::Vec2*)(&collapsedBVH[outIdx+2]))[offset] = ei::Vec2{boxMin_primCount.z, boxMax_primCost.z};
		((i32*)(&collapsedBVH[outIdx+3]))[offset] = node;
		((i32*)(&collapsedBVH[outIdx+3]))[offset+2] = ei::sum(extract_prim_counts<PrimCount_t<DescType>>(countCode));
	}
}

template < typename DescType >
__global__ void copy_to_collapsed_bvhD(
	const i32 numNodes,
	const i32 numInternalNodes,
	const ei::Vec4 * __restrict__ boundingBoxes,
	const i32* __restrict__ parents,
	const i32* __restrict__ offsets,
	ei::Vec4* collapsedBVH
) {
	i32 idx = threadIdx.x + blockIdx.x * blockDim.x + 1;

	if (idx >= numNodes)
		return;

	copy_to_collapsed_bvh<DescType>(boundingBoxes, parents, offsets, idx, numInternalNodes, collapsedBVH);
}


template < typename DescType >
void LBVHBuilder::build_lbvh(const DescType& desc,
							 const ei::Box& sceneBB,
							 const i32 numPrimitives
) {
	/*if(numPrimitives == 1) {
		// TODO remove this. 
		m_primIds.resize(1);
		m_primIds.synchronize<dev>();
		m_bvhNodes.resize(1);
		m_bvhNodes.synchronize<dev>();
		return;
	}*/

	const i32 numInternalNodes = numPrimitives - 1;
	const i32 numNodes = numInternalNodes + numPrimitives;


	i32 numBlocks, numThreads; // TODO: move to local scopes

	// Allocate memory for a part of the BVH.We do not know the final size yet and
	// cannot allocate the other parts in bvh.
	m_primIds.resize(numPrimitives * sizeof(i32));
	i32* primIds = as<i32>(m_primIds.acquire<DescType::DEVICE>());
	auto parents = make_udevptr_array<DescType::DEVICE, i32>(numNodes);

	// Allocate a block of temporary memory for several build purposes
	auto tmpMem = make_udevptr_array<DescType::DEVICE, u8>(ei::max(
		  numPrimitives * sizeof(MortonCode_t<DescType>) // For unsorted morton codes and
		+ numPrimitives * sizeof(i32),					 // primIds at the same time
		  numInternalNodes * sizeof(i32)				 // OR deviceCountes later
	));

	// Calculate Morton codes.
	{
		auto sortedMortonCodes = make_udevptr_array<DescType::DEVICE, MortonCode_t<DescType>>(numPrimitives);
		auto* mortonCodes = as<MortonCode_t<DescType>>(tmpMem.get());
		auto* primIdsUnsorted = as<i32>(tmpMem.get() + numPrimitives * sizeof(MortonCode_t<DescType>));
		if(DescType::DEVICE == Device::CUDA) {
			// Satisfy the compiler (this brach is never reached without having the correct type,
			// so the cast is effectless. However, the cleaner 'if constexpr' is not supported in cuda.
			//auto dobj = reinterpret_cast<const ObjectDescriptor<Device::CUDA>&>(obj);
			get_maximum_occupancy(numBlocks, numThreads, numPrimitives, calculate_morton_codesD<DescType>);
			calculate_morton_codesD <<< numBlocks, numThreads >>> (
				desc, sceneBB, numPrimitives, mortonCodes, primIdsUnsorted);
			cuda::check_error(cudaGetLastError());

			// Sort based on Morton codes.
			CuLib::DeviceSort(numPrimitives, mortonCodes, sortedMortonCodes.get(),
				primIdsUnsorted, primIds);
			cuda::check_error(cudaGetLastError());
		} else {
			for (i32 idx = 0; idx < numPrimitives; idx++)
			{
				sortedMortonCodes[idx] = calculate_morton_code<DescType>(desc, idx, sceneBB);
				primIds[idx] = idx;
			}

			// Sort based on Morton codes.
			thrust::sort_by_key(sortedMortonCodes.get(), mortonCodes + numPrimitives, primIds);
		}

		// Create BVH.
		// Layout: first internal nodes, then leves.
		if(DescType::DEVICE == Device::CUDA) {
			cudaFuncSetCacheConfig(build_lbvh_treeD<MortonCode_t<DescType>>, cudaFuncCachePreferL1);
			get_maximum_occupancy(numBlocks, numThreads, numInternalNodes, build_lbvh_treeD<MortonCode_t<DescType>>);
			build_lbvh_treeD<MortonCode_t<DescType>> <<< numBlocks, numThreads >>> (
				numPrimitives,
				sortedMortonCodes.get(),
				parents.get());
			cuda::check_error(cudaGetLastError());
		} else {
			for (i32 idx = 0; idx < numInternalNodes; idx++)
				build_lbvh_tree<MortonCode_t<DescType>>(numPrimitives,
					sortedMortonCodes.get(), parents.get(), idx);
		}
	}

	// Calculate bounding boxes and SAH.
	auto boundingBoxes = make_udevptr_array<DescType::DEVICE, ei::Vec4>(numNodes * 2);
	// The counters buffer is used with atomics to detect the order of executions within the launch.
	auto* deviceCounters = as<i32>(tmpMem.get());
	mem_set<DescType::DEVICE>(deviceCounters, 0xFF, numInternalNodes * sizeof(i32));
	// Allocate some memory of the later computation of partial BVH collapses.
	// This memory is initialized in the calculate_bounding_boxesD kernel to 1,
	// because setting 32bit integers to 1 is not possible with a memSet.
	auto collapseOffsets = make_udevptr_array<DescType::DEVICE, i32>(numInternalNodes);
	if(DescType::DEVICE == Device::CUDA) {
		// Calculate BVH bounding boxes.
		i32 bboxCacheSize = numThreads * sizeof(ei::Vec4) * 2; // TODO: numThreads is that of the last kernel launch, not of the comming one...
		cudaFuncSetCacheConfig(calculate_bounding_boxesD<DescType>, cudaFuncCachePreferShared);
		BoundingBoxFunctor functor;
		get_maximum_occupancy_variable_smem(numBlocks, numThreads, numPrimitives,
			calculate_bounding_boxesD<DescType>, functor);
		calculate_bounding_boxesD<<< numBlocks, numThreads, bboxCacheSize >>>(
			desc, numPrimitives, primIds, parents.get(),
			boundingBoxes.get(), deviceCounters,
			collapseOffsets.get()
		);
	} else {
		for(i32 idx = 0; idx < numPrimitives; idx++) {
			calculate_bounding_boxes(desc, idx, primIds[idx],
				numPrimitives, parents.get(), boundingBoxes.get(),
				deviceCounters, collapseOffsets.get());
		}
	}

	// Find out which nodes can be collapsed according to SAH.
	if(DescType::DEVICE == Device::CUDA) {
		get_maximum_occupancy(numBlocks, numThreads, numPrimitives, mark_nodesD<DescType>);
		mark_nodesD<DescType> <<< numBlocks, numThreads >>>(numInternalNodes,
			boundingBoxes.get(), parents.get(),
			collapseOffsets.get()
		);
	} else {
		for(i32 idx = 0; idx < numPrimitives; idx++) {
			mark_collapsed_nodes<DescType>(boundingBoxes.get(), parents.get(),
				idx + numInternalNodes, collapseOffsets.get());
		}
	}

	// Scan to get values for offsets.
	i32 numRemovedInternalNodes;
	if(DescType::DEVICE == Device::CUDA) {
		CuLib::DeviceInclusiveSum(numInternalNodes, collapseOffsets.get(), collapseOffsets.get());
		copy(&numRemovedInternalNodes, collapseOffsets.get() + numInternalNodes - 1, sizeof(i32));
		numRemovedInternalNodes = numInternalNodes - numRemovedInternalNodes;
		// Scan to get number of leaves arised from internal nodes before current node.
//		CuLib::DeviceExclusiveSum(numInternalNodes + 1, leafMarks, leafMarks);
	} else {
		// Scan to get values for offsets.
		thrust::inclusive_scan(collapseOffsets.get(), collapseOffsets.get()+numInternalNodes, collapseOffsets.get());
		numRemovedInternalNodes = numInternalNodes - collapseOffsets[numInternalNodes - 1];
		// Scan to get number of leaves arised from internal nodes before current node.
		//thrust::exclusive_scan(leafMarks, leafMarks + numInternalNodes + 1, leafMarks);
	}

	logInfo("[LBVHBuilder::build_lbvh] collapsing removed ", numRemovedInternalNodes, " nodes.");

	// Write the final compacted BVH
	i32 numFloat4InCollapsedBVH = numInternalNodes - numRemovedInternalNodes;
	m_bvhNodes.resize(numFloat4InCollapsedBVH * sizeof(ei::Vec4));
	ei::Vec4* collapsedBVH = as<ei::Vec4>(m_bvhNodes.acquire<DescType::DEVICE>());
	if(DescType::DEVICE == Device::CUDA) {
		get_maximum_occupancy(numBlocks, numThreads, numNodes, copy_to_collapsed_bvhD<DescType>);
		copy_to_collapsed_bvhD<DescType> <<< numBlocks, numThreads >>>(
			numNodes, numInternalNodes, boundingBoxes.get(), parents.get(),
			collapseOffsets.get(), collapsedBVH);
	} else {
		for(i32 idx = 0; idx < numNodes; ++idx)
			copy_to_collapsed_bvh<DescType>(
				boundingBoxes.get(), parents.get(), collapseOffsets.get(),
				idx, numInternalNodes, collapsedBVH);
	}
}

template < Device dev >
void LBVHBuilder::build(ObjectDescriptor<dev>& obj, const ei::Box& sceneBB) {
	build_lbvh<ObjectDescriptor<dev>>(obj, sceneBB, obj.numPrimitives);
	m_primIds.mark_changed(dev);
	m_bvhNodes.mark_changed(dev);
}

template void LBVHBuilder::build<Device::CPU>(ObjectDescriptor<Device::CPU>&, const ei::Box&);
template void LBVHBuilder::build<Device::CUDA>(ObjectDescriptor<Device::CUDA>&, const ei::Box&);

template < Device dev >
void LBVHBuilder::build(
	const SceneDescriptor<dev>& scene
) {
	build_lbvh<SceneDescriptor<dev>>(scene, scene.aabb, scene.numInstances);
	m_primIds.mark_changed(dev);
	m_bvhNodes.mark_changed(dev);
}

template void LBVHBuilder::build<Device::CPU>(const SceneDescriptor<Device::CPU>&);
template void LBVHBuilder::build<Device::CUDA>(const SceneDescriptor<Device::CUDA>&);

}}} // namespace mufflon
