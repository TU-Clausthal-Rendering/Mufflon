#pragma once

#include "core/math/intersection_areas.hpp"
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
	static constexpr float SPLIT_FACTOR = 16.f;
	// At some time the counting should stop -- otherwise the counter will overflow inevitable.
	static constexpr int FILL_ITERATIONS = 1000;
public:
	DmOctree(const ei::Box& sceneBounds, int capacity) {
		// Slightly enlarge the volume to avoid numerical issues on the boundary
		ei::Vec3 sceneSize = (sceneBounds.max - sceneBounds.min) * 1.002f;
		m_sceneSize = sceneSize;
		m_sceneSizeInv = 1.0f / sceneSize;
		m_minBound = sceneBounds.min - sceneSize * (0.001f / 1.002f);
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
	}

	// Initialize iteration count dependent data (1-indexed).
	// The first iteration must call set_iteration(1);
	void set_iteration(int iter) {
		int iterClamp = ei::min(FILL_ITERATIONS, iter);
		m_stopFilling = iter > FILL_ITERATIONS;
		m_densityScale = 1.0f / iterClamp;
		m_splitCountDensity = ei::ceil(SPLIT_FACTOR * 8) * iterClamp;
		// Set the counter of all unused cells to the number of expected samples
		// times 2. A planar surface will never extend to all eight cells. It might
		// intersect 7 of them, but still the distribution is one of a surface.
		// Therefore, the factor 2 (distribute among 4 cells) gives a much better initial
		// value.
		if(!m_stopFilling)
			for(int i = m_allocationCounter.load(); i < m_capacity; ++i)
				m_nodes[i].store(ei::ceil(SPLIT_FACTOR * 2 * iter));
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
		i8 lvlBuffer[16];
		int* parents = buffer;
		int* current = buffer + 8;
		i8* parentLvl = lvlBuffer;
		i8* currentLvl = lvlBuffer + 8;
		for(int i=0; i<8; ++i) {
			current[i] = 0;	// Initialize to root level
			currentLvl[i] = 0;
		}
		int lvl = 0;
		ei::IVec3 parentMinPos { 0 };
		bool anyHadChildren = m_nodes[0].load() < 0;
		while(anyHadChildren) {
			++lvl;
			std::swap(parents, current);
			std::swap(parentLvl, currentLvl);
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
					cellPos = clamp(cellPos, 0, lvlRes - 1);
					int localChildIdx = (cellPos.x & 1) + 2 * (cellPos.y & 1) + 4 * (cellPos.z & 1);
					current[i] = -c + localChildIdx;
					currentLvl[i] = lvl;
				} else { // Otherwise copy the parent to the next finer level.
					current[i] = parentAddress;
					currentLvl[i] = parentLvl[parentIdx];
				}
			}
			parentMinPos = lvlPos;
			// Check if any of the current nodes has children -> must proceed
			anyHadChildren = false;
			for(int i = 0; i < 8; ++i)
				anyHadChildren |= m_nodes[current[i]].load() < 0;
		}
		// The loop terminates if all 8 cells in 'current' are leaves.
		// This means we want to interpolate 'current' on 'lvl'.
		int lvlRes = 1 << lvl;
		ei::Vec3 tPos = normPos * lvlRes - 0.5f;
		ei::IVec3 gridPos = ei::floor(tPos);
		//return m_nodes[current[0]].load() * m_densityScale;
		ei::Vec3 ws[2];
		ws[1] = tPos - gridPos;
		ws[0] = 1.0f - ws[1];
		float countSum = 0.0f, areaSum = 0.0f, wSum = 0.0f;
		float densitySum = 0.0f;
		ei::Vec3 cellSize { m_sceneSize / lvlRes };
		for(int i = 0u; i < 8u; ++i) {
			int ix = i & 1, iy = (i>>1) & 1, iz = i>>2;
			ei::Vec3 localPos = ws[1] + 0.5f - ei::IVec3{ix, iy, iz};
			localPos *= cellSize;
			float area = math::intersection_area_nrm(cellSize, localPos, normal);
			// Compute trilinear interpolated result of count and area (running sum)
			float w = ws[ix].x * ws[iy].y * ws[iz].z;
			mAssert(m_nodes[current[i]].load() >= 0);
			if(area > 0.0f) {
				float lvlFactor = float(1 << ((lvl - currentLvl[i]) * 2));
				float lvlFactorA = 1.0f;//float(1 << ((lvl - currentLvl[i]) * 2));
				countSum += m_nodes[current[i]].load() * w / lvlFactor;
				areaSum += area * w / lvlFactorA;
				densitySum += m_nodes[current[i]].load() * w / (lvlFactor * area);
				wSum += w;
			}
		}
		mAssert(areaSum > 0.0f);
		return sdiv(countSum, areaSum) * m_densityScale;
		//return sdiv(densitySum, wSum) * m_densityScale;
	}

	int capacity() const { return m_capacity; }
	int size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
	// Get the size of the associated memory excluding this instance.
	std::size_t mem_size() const { return sizeof(std::atomic_int32_t) * m_capacity; }
private:
	float m_densityScale;		// 1/#iterations to normalize the counters into a density
	int m_splitCountDensity;	// The number when a node is split must be a multiple of 8 and must grow proportional to #iterations
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
};

}} // namespace mufflon::data_structs
