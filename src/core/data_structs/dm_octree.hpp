#pragma once

#include "core/math/intersection_areas.hpp"
#include "util/log.hpp"
#include <ei/3dtypes.hpp>
#include <memory>
#include <atomic>

namespace mufflon { namespace data_structs {

template<typename T>
inline void atomic_max(std::atomic<T>& a, T b) {
	T oldV = a.load();
	while(oldV < b && !a.compare_exchange_weak(oldV, b)) ;
}

// A sparse octree with atomic insertion to measure the density of elements in space.
class DmOctree {
	// At some time the counting should stop -- otherwise the counter will overflow inevitable.
	static constexpr int FILL_ITERATIONS = 1000;
public:
	// splitFactor: Number of photons in one cell (per iteration) before it is splitted.
	DmOctree(const ei::Box& sceneBounds, int capacity, float splitFactor) {
		if(splitFactor <= 1.0f)
			logError("[DmOctree] Split factor must be larger than 1. Otherwise the tree will be split infinitely. Setting to 1.1 instead ", splitFactor);
		// Slightly enlarge the volume to avoid numerical issues on the boundary
		ei::Vec3 sceneSize = (sceneBounds.max - sceneBounds.min) * 2.002f;
		m_sceneSize = sceneSize;
		m_sceneSizeInv = 1.0f / sceneSize;
		m_minBound = sceneBounds.min - sceneSize * (2.002f - 1.0f) / 2.0f;
		m_capacity = 1 + ((capacity + 7) & (~7));
		m_nodes = std::make_unique<std::atomic_int32_t[]>(m_capacity);;
		// Root nodes have a count of 0
		m_allocationCounter.store(1);
		m_nodes[0].store(0);
		// TODO: parallelize?
		// The other nodes are only used if the parent is split
		//for(int i = 1; i < m_capacity; ++i)
		//	m_nodes[i].store(ei::ceil(SPLIT_FACTOR));
		m_depth.store(0);
		m_splitFactor = ei::max(1.1f, splitFactor);
	}

	// Initialize iteration count dependent data (1-indexed).
	// The first iteration must call set_iteration(1);
	void set_iteration(int iter) {
		int iterClamp = ei::min(FILL_ITERATIONS, iter);
		m_stopFilling = iter > FILL_ITERATIONS;
		m_densityScale = 1.0f / iterClamp;
		m_splitCountDensity = ei::ceil(m_splitFactor) * iterClamp;
		// Set the counter of all unused cells to the number of expected samples
		// times 2. A planar surface will never extend to all eight cells. It might
		// intersect 7 of them, but still the distribution is one of a surface.
		// Therefore, the factor 2 (distribute among 4 cells) gives a much better initial
		// value.
		if(!m_stopFilling)
			for(int i = m_allocationCounter.load(); i < m_capacity; ++i)
				m_nodes[i].store(ei::ceil(iter * m_splitFactor / 4));
	}

	// Overwrite all counters with 0, but keep allocation and child pointers.
	void clear_counters() {
		int n = m_allocationCounter.load();
		for(int i = 0; i < n; ++i)
			if(m_nodes[i].load() > 0)
				m_nodes[i].store(0);
	}

	// Clear entire structure
	void clear() {
		m_allocationCounter.store(1);
		m_nodes[0].store(0);
		m_depth.store(0);
	}

	void increase_count(const ei::Vec3& pos) {
		if(m_stopFilling) return;
		ei::Vec3 normPos = (pos - m_minBound) * m_sceneSizeInv;
		int countOrChild = increment_if_positive(0);
		countOrChild = split_node_if_necessary(0, countOrChild, 0);
		int edgeL = 1;
		int currentDepth = 0;
		while(countOrChild < 0) {
			edgeL *= 2;
			++currentDepth;
			// Get the relative index of the child [0,7]
			ei::IVec3 intPos = (ei::IVec3{ normPos * edgeL }) & 1;
			int idx = intPos.x + 2 * (intPos.y + 2 * intPos.z);
			idx -= countOrChild;	// 'Add' global offset (which is stored negative)
			countOrChild = increment_if_positive(idx);
			//if(currentDepth == 3) break;
			countOrChild = split_node_if_necessary(idx, countOrChild, currentDepth);
		}
	}

	// Idea for truly smooth interpolation:
	//	Track the center positions of the cells to interpolate. Those might be
	//	the same for multiple coordinates. The shape formed by these centers is
	//	a convex polyhedron. Then one needs a kind of general barycentric coordinates
	//	to interpolate within this polyhedron:
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf (with pseudocode)
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.9919&rep=rep1&type=pdf
	//		* 2D: http://geometry.caltech.edu/pubs/MHBD02.pdf
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal) {
		ei::Vec3 offPos = pos - m_minBound;
		ei::Vec3 normPos = offPos * m_sceneSizeInv;
		// Get the integer position on the finest level.
		int gridRes = 1 << m_depth.load();
		ei::IVec3 iPos { normPos * gridRes };
		// Get root value. This will most certainly be a child pointer...
		int countOrChild = m_nodes[0].load();
		// The most significant bit in iPos distinguishes the children of the root node.
		// For each level, the next bit will be the relevant one.
		int currentLvlMask = gridRes;
		while(countOrChild < 0) {
			currentLvlMask >>= 1;
			// Get the relative index of the child [0,7]
			int idx = ((iPos.x & currentLvlMask) ? 1 : 0)
					+ ((iPos.y & currentLvlMask) ? 2 : 0)
					+ ((iPos.z & currentLvlMask) ? 4 : 0);
			// 'Add' global offset (which is stored negative)
			idx -= countOrChild;
			countOrChild = m_nodes[idx].load();
		}
		if(countOrChild > 0) {
			// Get the world space cell boundaries
			int currentGridRes = gridRes / currentLvlMask;
			ei::IVec3 cellPos = iPos / currentLvlMask;
			ei::Vec3 cellSize = 1.0f / (currentGridRes * m_sceneSizeInv);
			ei::Vec3 cellMin = cellPos * cellSize;
			//ei::Vec3 cellMax = cellMin + cellSize;
			//float area = math::intersection_area(cellMin, cellMax, offPos, normal);
			float area = math::intersection_area_nrm(cellSize, offPos - cellMin, normal);
			return sdiv(m_densityScale * countOrChild, area);
			//return m_densityScale * countOrChild;
		}
		return 0.0f;
	}

	float get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal) {
		ei::Vec3 offPos = pos - m_minBound;
		ei::Vec3 normPos = offPos * m_sceneSizeInv;
		// Get the integer position on the finest level.
		int maxLvl = m_depth.load();
		int gridRes = 1 << (maxLvl + 1);
		ei::IVec3 iPos { normPos * gridRes };
		// Memory to track nodes
		int buffer[16];
		float areaBuffer[16];
		int* parents = buffer;
		int* current = buffer + 8;
		float* parentArea = areaBuffer;
		float* currentArea = areaBuffer + 8;
		for(int i=0; i<8; ++i) {
			current[i] = 0;	// Initialize to root level
			currentArea[i] = -1;
		}
		int lvl = 0;
		ei::IVec3 parentMinPos { 0 };
		bool anyHadChildren = m_nodes[0].load() < 0;
		while(anyHadChildren) {
			++lvl;
			std::swap(parents, current);
			std::swap(parentArea, currentArea);
			// Compute the 8 nodes which are required for the bilinear interpolation.
			// Interpolation jumps at half cells -> need index from next level
			// and a remapping.
			// Cell index:       0       1       2       3
			//               ├───┬───┼───┬───┼───┬───┼───┬───┤
			// next-lvl        0   1   2   3   4   5   6   7
			// Min index      -1   0   0   1   1   2   2   3
			ei::IVec3 nextLvlPos = iPos >> (maxLvl - lvl);	// Next level coordinate
			ei::IVec3 lvlPos = nextLvlPos / 2 - 1 + (nextLvlPos & 1);	// Min coordinate of the 8 cells on next level
			int lvlRes = 1 << lvl;
			for(int i = 0; i < 8; ++i) {
				ei::IVec3 cellPos = lvlPos + ei::IVec3{ i & 1, (i >> 1) & 1, i >> 2 };
				// We need to find the parent in the 'parents' buffer array.
				// Since the window of interpolation moves the reference coordinate
				// we subtract 'parentMinPos' scaled to the current level.
				ei::IVec3 localParent = (cellPos - parentMinPos * 2) / 2;
				mAssert(localParent >= 0 && localParent <= 1);
				//ei::IVec3 localParent = 1 - (clamp(cellPos, 1, lvlRes-2) & 1);
				//ei::IVec3 localParent = 1 - (cellPos & 1);
				int parentIdx = localParent.x + 2 * localParent.y + 4 * localParent.z;
				// Check if parent node has children.
				int parentAddress = parents[parentIdx];
				int c = m_nodes[parentAddress].load();
				if(c < 0) {
					// Insert the child node's address
					int localChildIdx = (cellPos.x & 1) + 2 * (cellPos.y & 1) + 4 * (cellPos.z & 1);
					current[i] = -c + localChildIdx;
					currentArea[i] = -1.0f;
				} else { // Otherwise copy the parent to the next finer level.
					current[i] = parentAddress;
					currentArea[i] = parentArea[parentIdx];
				}
			}
			parentMinPos = lvlPos;
			// Check if any of the current nodes has children -> must proceed.
			// Also, compute the areas of leaf nodes.
			anyHadChildren = false;
			const ei::Vec3 cellSize = m_sceneSize / lvlRes;
			const float avgArea = (cellSize.x * cellSize.y + cellSize.x * cellSize.z + cellSize.y * cellSize.z) / 3.0f;
			for(int i = 0; i < 8; ++i) {
				const int c = m_nodes[current[i]].load();
				anyHadChildren |= c < 0;
				// Density not yet computed? Density might have been copied
				// from parent in which case we do not need to compute it anew.
				if(c >= 0 && currentArea[i] < 0.0f) {
					const int ix = i & 1, iy = (i>>1) & 1, iz = i>>2;
					const ei::Vec3 localPos = offPos - (lvlPos + ei::IVec3{ix, iy, iz}) * cellSize;
					float area = math::intersection_area_nrm(cellSize, localPos, normal);
/*					float area2 = math::intersection_area_nrm(cellSize * 1.02f, localPos + cellOffset, normal);
					ei::Vec3 cellMin = (lvlPos + ei::IVec3{ix, iy, iz}) * m_sceneSize / lvlRes;
					ei::Vec3 cellMax = cellMin + m_sceneSize / lvlRes;
					float area3 = math::intersection_area(cellMin, cellMax, offPos, normal);
					float area4 = math::intersection_area(cellMin - cellOffset, cellMax + cellOffset, offPos, normal);*/
					currentArea[i] = area;//ei::lerp(avgArea, area, 0.9f);
					//currentArea[i] = math::intersection_area_nrm(cellSize, offPos - (lvlPos + 0.5f) * cellSize, normal);
					//currentArea[i] = math::intersection_area_nrm(cellSize * 2, offPos - lvlPos * cellSize, normal) / 4.0f;
				}
			}
		}
		// The loop terminates if all 8 cells in 'current' are leaves.
		// This means we want to interpolate 'current' on 'lvl'.
		const int lvlRes = 1 << lvl;
		ei::Vec3 tPos = normPos * lvlRes - 0.5f;
		ei::IVec3 gridPos = ei::floor(tPos);
		//return m_nodes[current[0]].load() * m_densityScale;
		ei::Vec3 ws[2];
		ws[1] = tPos - gridPos;
		ws[0] = 1.0f - ws[1];
		float countSum = 0.0f, areaSum = 0.0f;
		const ei::Vec3 cellSize { m_sceneSize / lvlRes };
		const float avgArea = (cellSize.x * cellSize.y + cellSize.x * cellSize.z + cellSize.y * cellSize.z) / 3.0f;
		for(int i = 0; i < 8; ++i)
		{
			const int ix = i & 1, iy = (i>>1) & 1, iz = i>>2;
			const ei::Vec3 localPos = offPos - (gridPos + ei::IVec3{ix, iy, iz}) * cellSize;
			float area = math::intersection_area_nrm(cellSize, localPos, normal);
			float areaReg = ei::lerp(avgArea, area, 0.9f);
			// Compute trilinear interpolated result of count and area (running sum)
			float w = ws[ix].x * ws[iy].y * ws[iz].z;
			mAssert(m_nodes[current[i]].load() >= 0);
			//if(area > 0.0f)
			if(area > 0.0f && currentArea[i] > 0.0f)
			{
				//ei::Vec3 cellMin = (gridPos + ei::IVec3{ix, iy, iz}) * m_sceneSize / lvlRes;
				//ei::Vec3 cellMax = cellMin + m_sceneSize / lvlRes;
				//if(currentArea[i] == 0.0f) __debugbreak();
				//float lvlFactor = area / currentArea[i];
				//float lvlFactor = ei::max(area, avgArea * 0.1f) / ei::max(currentArea[i], avgArea * 0.1f);
				float lvlFactor = (area + avgArea * 0.01f) / (currentArea[i] + avgArea * 0.01f);
				//float lvlFactor = avgArea / currentArea[i];
				//float lvlFactor = 1.0f / currentArea[i];
				countSum += m_nodes[current[i]].load() * w * lvlFactor;
				//areaSum += area / 4.0f;
				//areaSum += area * w;
				areaSum += area * w;
				//areaSum += w;
			}
		}
		//areaSum = math::intersection_area_nrm(cellSize, offPos - (gridPos + 0.5f) * cellSize, normal);
		//areaSum = math::intersection_area_nrm(cellSize * 2, offPos - gridPos * cellSize, normal) / 4.0f;
		mAssert(areaSum > 0.0f);
		return sdiv(countSum, areaSum) * m_densityScale;
	}

	void balance(int current = 0, int nx = 0, int ny = 0, int nz = 0, int px = 0, int py = 0, int pz = 0) {
		// Call balance for each child recursively, if the child has children itself.
		// Otherwise balance is satisfied.
		int children = -m_nodes[current].load();
		if(children <= 0) return;	// No tree here
		for(int i = 0; i < 8; ++i) {
			int childC = m_nodes[children + i].load();
			if(childC < 0) {
				// To make the recursive call we need all the neighbors on the child-level.
				// If they do not extist we need to split the respective cell.
				const ei::IVec3 localPos { i & 1, (i>>1) & 1, i>>2 };
				int cnx = find_neighbor(localPos, 0, 0, nx, children);
				int cny = find_neighbor(localPos, 1, 0, ny, children);
				int cnz = find_neighbor(localPos, 2, 0, nz, children);
				int cpx = find_neighbor(localPos, 0, 1, px, children);
				int cpy = find_neighbor(localPos, 1, 1, py, children);
				int cpz = find_neighbor(localPos, 2, 1, pz, children);
				balance(children + i, cnx, cny, cnz, cpx, cpy, cpz);
			}
		}
	}

	// dir: 0 search in negative dir, 1 search in positive dir
	int find_neighbor(const ei::IVec3& localPos, int dim, int dir, int parentNeighbor, int siblings) {
		int cn = 0;	// Initialize to outer boundary
		// The adoint position is the child index of the neighbor (indepndent of the parent).
		// It merely flips the one coordinate of the relevant dimension
		int adjointIdx = 0;
		adjointIdx += (dim == 0 ? 1-localPos[0] : localPos[0]);
		adjointIdx += (dim == 1 ? 1-localPos[1] : localPos[1]) * 2;
		adjointIdx += (dim == 2 ? 1-localPos[2] : localPos[2]) * 4;
		if(localPos[dim] == dir && parentNeighbor > 0) { // Not on boundary
			int nC = m_nodes[parentNeighbor].load();
			if(nC >= 0) nC = split(parentNeighbor);
			cn = -nC + adjointIdx;
		} else if(localPos[dim] == (1-dir)) {
			cn = siblings + adjointIdx;
		}
		return cn;
	}

	int capacity() const { return m_capacity; }
	int size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
	// Get the size of the associated memory excluding this instance.
	std::size_t mem_size() const { return sizeof(std::atomic_int32_t) * m_capacity; }
private:
	float m_densityScale;		// 1/#iterations to normalize the counters into a density
	float m_splitFactor;		// Number of photons per iteration before a cell is split
	int m_splitCountDensity;	// Total number of photons before split
	ei::Vec3 m_minBound;
	ei::Vec3 m_sceneSizeInv;
	ei::Vec3 m_sceneSize;
	// Nodes consist of 8 atomic counters OR child indices. Each number is either a
	// counter (positive) or a negated child index.
	std::unique_ptr<std::atomic_int32_t[]> m_nodes;
	std::atomic_int32_t m_allocationCounter;
	std::atomic_int32_t m_depth;
	int m_capacity;
	bool m_stopFilling;

	// Returns the new value
	int increment_if_positive(int idx) {
		int oldV = m_nodes[idx].load();
		int newV;
		do {
			if(oldV < 0) return oldV;	// Do nothing, the value is a child pointer
			newV = oldV + 1;			// Increment
		} while(!m_nodes[idx].compare_exchange_weak(oldV, newV));	// Write if nobody destroyed the value
		return newV;
	}

	// Returns the new child pointer or 0
	int split_node_if_necessary(int idx, int count, int currentDepth) {
		// The node must be split if its density gets too high
		if(count >= m_splitCountDensity) {
			// Only one thread is responsible to do the allocation
			if(count == m_splitCountDensity) {
				int child = m_allocationCounter.fetch_add(8);
				if(child >= m_capacity) { // Allocation overflow
					m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
					return 0;
				}
				// We do not know anything about the distribution of of photons -> equally
				// distribute. Therefore, all eight children are initilized with SPLIT_FACTOR on clear().
				m_nodes[idx].store(-child);
				// Update depth
				atomic_max(m_depth, currentDepth+1);
				// The current photon is already counted before the split -> return stop
				return 0;
			} else {
				// Spin-lock until the responsible thread has set the child pointer
				int child = m_nodes[idx].load();
				while(child > 0) {
					// Check for allocation overflow
					if(m_allocationCounter.load() > m_capacity)
						return 0;
					child = m_nodes[idx].load();
				}
				return child;
			}
		}
		return count;
	}

	// Non-atomic unconditional split. Returns the new address
	int split(int idx) {
		int child = m_allocationCounter.fetch_add(8);
		if(child >= m_capacity) { // Allocation overflow
			m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
			return 0;
		}
		m_nodes[idx].store(-child);
		return -child;
	}
};

}} // namespace mufflon::data_structs
