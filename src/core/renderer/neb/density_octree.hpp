#pragma once

#include "core/math/intersection_areas.hpp"
#include <ei/3dtypes.hpp>
#include <memory>
#include <atomic>

namespace mufflon::renderer {

	template<typename T>
	inline void atomic_max(std::atomic<T>& a, T b) {
		T oldV = a.load();
		while(oldV < b && !a.compare_exchange_weak(oldV, b)) ;
	}

	// A sparse octree with atomic insertion to measure the density of elements in space.
	class DensityOctree {
		static constexpr float SPLIT_FACTOR = 16.f;
		// At some time the counting should stop -- otherwise the counter will overflow inevitable.
		static constexpr int FILL_ITERATIONS = 1000;
	public:
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

		void initialize(const ei::Box& sceneBounds, int capacity) {
			// Slightly enlarge the volume to avoid numerical issues on the boundary
			ei::Vec3 sceneSize = (sceneBounds.max - sceneBounds.min) * 1.002f;
			m_sceneSizeInv = 1.0f / sceneSize;
			m_sceneScale = len(sceneSize);
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

		// Overwrite all counters with 0, but keep allocation and child pointers.
		void clear_counters() {
			int n = m_allocationCounter.load();
			for(int i = 0; i < n; ++i)
				if(m_nodes[i].load() > 0)
					m_nodes[i].store(0);
		}

		void increment(const ei::Vec3& pos) {
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
				countOrChild = split_node_if_necessary(idx, countOrChild, currentDepth);
			}
		}

		float get_density(const ei::Vec3& pos, const ei::Vec3& normal, float* size = nullptr) {
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
				float area = math::intersection_area_nrm(cellSize, offPos - cellMin, normal);
				// Sometimes the above method returns zero. Therefore we restrict the
				// area to something larger then a hunderds part of an approximate cell area.
				float cellDiag = m_sceneScale / currentGridRes;
				float minArea = cellDiag * cellDiag;
				if(size) { *size = cellDiag; minArea *= 0.1f; }
				else minArea *= 0.01f;
				return m_densityScale * countOrChild / ei::max(minArea, area);
			}
			return 0.0f;
		}

		float get_density_robust(const ei::Vec3& pos, const scene::TangentSpace& ts) {
			float d[5];
			float cellDiag = 1e-3f;
			int count = 0;
			d[0] = get_density(pos, ts.geoN, &cellDiag);
			cellDiag *= 1.1f;
			if(d[0] > 0.0f) ++count;
			d[count] = get_density(pos + ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos - ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos + ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos - ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			// Find the median via selection sort up to the element m.
			// Prefer the greater element, because overestimations do not
			// cause such visible artifacts.
			int m = count / 2;
			for(int i = 0; i <= m; ++i) for(int j = i+1; j < count; ++j)
				if(d[j] < d[i])
					std::swap(d[i], d[j]);
			return d[m];
			/*float maxD = d[0];
			for(int i = 1; i < count; ++i)
				if(d[i] > maxD) maxD = d[i];
			return maxD;*/
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
		// Nodes consist of 8 atomic counters OR child indices. Each number is either a
		// counter (positive) or a negated child index.
		std::unique_ptr<std::atomic_int32_t[]> m_nodes;
		std::atomic_int32_t m_allocationCounter;
		std::atomic_int32_t m_depth;
		int m_capacity;
		float m_sceneScale;
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


	// A sparse octree with atomic insertion to measure the density of elements in space.
	// Keeps count and child indices split
	class SplitDensityOctree {
		static constexpr float SPLIT_FACTOR = 16.0f;
		// At some time the counting should stop -- otherwise the counter will overflow inevitable.
		static constexpr int FILL_ITERATIONS = 1000;
		static constexpr u32 MAX_TREE_DEPTH = 32;

	public:
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
				for(u32 i = m_allocationCounter.load(); i < m_capacity; ++i) {
					m_cellCounter[i].store(ei::ceil(SPLIT_FACTOR * 2 * iter));
					m_cellChildIndices[i].store(0);
				}
		}

		void initialize(const ei::Box& sceneBounds, int capacity) {
			// Slightly enlarge the volume to avoid numerical issues on the boundary
			ei::Vec3 sceneSize = (sceneBounds.max - sceneBounds.min) * 1.002f;
			m_sceneSizeInv = 1.0f / sceneSize;
			m_sceneScale = len(sceneSize);
			m_minBound = sceneBounds.min - sceneSize * (0.001f / 1.002f);
			m_capacity = 1 + ((capacity + 7) & (~7));
			m_cellCounter = std::make_unique<std::atomic_uint32_t[]>(m_capacity);
			m_cellChildIndices = std::make_unique<std::atomic_uint32_t[]>(m_capacity);
			// Root nodes have a count of 0
			m_allocationCounter.store(1);
			m_cellCounter[0].store({});
			m_cellChildIndices[0].store(0);
			// TODO: parallelize?
			// The other nodes are only used if the parent is split
			//for(int i = 1; i < m_capacity; ++i)
			//	m_nodes[i].store(ei::ceil(SPLIT_FACTOR));
			m_depth.store(0);
		}

		// Overwrite all counters with 0, but keep allocation and child pointers.
		void clear_counters() {
			u32 n = m_allocationCounter.load();
			for(u32 i = 0; i < n; ++i)
				m_cellCounter[i].store({});
		}

		void increment(const ei::Vec3& pos) {
			if(m_stopFilling) return;
			ei::Vec3 normPos = (pos - m_minBound) * m_sceneSizeInv;
			u32 childIdx = m_cellChildIndices[0];
			u32 count = ++m_cellCounter[0];
			childIdx = split_node_if_necessary(0, childIdx, count, 0);
			u32 edgeL = 1;
			u32 currentDepth = 0;
			while(childIdx != 0) {
				edgeL *= 2;
				++currentDepth;
				// Get the relative index of the child [0,7]
				ei::UVec3 intPos = (ei::UVec3{ normPos * edgeL }) & 1;
				u32 idx = intPos.x + 2 * (intPos.y + 2 * intPos.z);
				idx += childIdx;	// Add global offset

				childIdx = m_cellChildIndices[idx];
				u32 count = ++m_cellCounter[idx];
				childIdx = split_node_if_necessary(idx, childIdx, count, currentDepth);
			}
		}

		float get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal, float* size = nullptr) {
			// The expected density of a cell's child is 2/8 cells are filled
			constexpr int CELL_COUNT = 8;
			constexpr float EXPECTED_DENSITY_FRAC = 2.f / 8.f;
			// Local "pair" to keep index and count for a cell close together
			struct CellInfo {
				u32 globalIndex;
				float count;
			};

			const ei::Vec3 offPos = pos - m_minBound;
			const ei::Vec3 normPos = offPos * m_sceneSizeInv;

			// Special case: only root level is ignored since it is HIGHLY likely that a sublevel exists
			// The indices are ordered in cube order: topleft first, bottom right last
			// We start with the 2x2x2 cube, but cellsPerDim tracks the size for the look-ahead
			u32 cellsPerDim = 2u << 2u;
			CellInfo currCellInfo[CELL_COUNT];
			// The initial cells are the root's children (1, ..., 8)
			for(u32 i = 0u; i < CELL_COUNT; ++i)
				currCellInfo[i] = { i + 1, static_cast<float>(m_cellCounter[i + 1]) };

			// Tracks at what offset our current cell cube is on the grid resolution we use to determine
			// which cells we use as parents (see nextCoord in the loop)
			ei::UVec3 globalGridOffset{ 0 };
			// Termination criterion for the traversal is that no cell of our cube
			// has children anymore
			bool hadRefinement = true;

			while(hadRefinement) {
				// The next coordinate will tell us how we need to shift our interpolation cube
				const ei::UVec3 nextCoords = ei::UVec3{ normPos * cellsPerDim } - globalGridOffset;
				// Determine the next actual cell indices
				// Next index: 0-2 -> 0, 3-4 -> 1, 5-7 -> 2
				const ei::UVec3 nextGridRelIndices{
					nextCoords.x <= 2u ? 0u : (nextCoords.x <= 4u ? 1u : 2u),
					nextCoords.y <= 2u ? 0u : (nextCoords.y <= 4u ? 1u : 2u),
					nextCoords.z <= 2u ? 0u : (nextCoords.z <= 4u ? 1u : 2u)
				};
				// The offset needs to be calculated for the grid resolution in the next iteration
				// and adjusted the 'next next' grid resolution (matching nextCoords)
				globalGridOffset += (nextGridRelIndices << 1u);
				globalGridOffset <<= 1u;
				//globalGridOffset += (globalGridOffset + (nextGridRelIndices << 1)) << 1;

				// Compute the global indices (if cells actually exist)
				hadRefinement = false;
				CellInfo nextCellInfo[CELL_COUNT];
				for(u32 i = 0u; i < CELL_COUNT; ++i) {
					// The x/y/z of our current cube combined with the next cell indices give us
					// the actual grid coordinates locally in our cube
					const ei::UVec3 localCoord{ i & 1u, (i & 2u) >> 1u, (i & 4u) >> 2u };
					const ei::UVec3 gridCoord = localCoord + nextGridRelIndices;
					const ei::UVec3 parentCoord = gridCoord >> 1;
					const int parentIdx = parentCoord.x + 2u * (parentCoord.y + 2u * parentCoord.z);
					// Check if the parent cell existed or if we interpolated the value already
					const int parentGlobalIdx = currCellInfo[parentIdx].globalIndex;
					if(parentGlobalIdx != 0u) {
						// Check if the parent cell has children of which we can get the exact density
						const u32 childIdx = m_cellChildIndices[parentGlobalIdx];
						if(childIdx == 0u) {
							// No children: interpolate value from parents; for this we need the
							// interpolation factors which we obtain from the child coordinates (gridCoord inside):
							//     -0.5 -0.25  0   0.25 0.5 0.75   1  1.25  1.5
							//  -0.5 ┌─────────┬─────────┬─────────┬─────────┐
							//       │         │         │         │         │
							// -0.25 │    0    │    1    │    2    │    3    │
							//       │         │         │         │         │
							//  0    ├─────────┼─────────┼─────────┼─────────┤
							//       │         │         │         │         │
							// 0.25  │    4    │    5    │    6    │    7    │
							//       │         │         │         │         │
							const ei::Vec3 interpFactors = ei::clamp(gridCoord * 0.5f - 0.25f, 0.f, 1.f);
							const float densityZ0 = ei::bilerp(currCellInfo[0].count, currCellInfo[1].count,
															   currCellInfo[2].count, currCellInfo[3].count,
															   interpFactors.x, interpFactors.y);
							const float densityZ1 = ei::bilerp(currCellInfo[4].count, currCellInfo[5].count,
															   currCellInfo[6].count, currCellInfo[7].count,
															   interpFactors.x, interpFactors.y);
							const float interpDensity = ei::lerp(densityZ0, densityZ1, interpFactors.z);

							// Use the expected density distribution to determine the fraction that
							// makes it to the child, which is quarter coverage
							nextCellInfo[i] = { 0u, interpDensity * EXPECTED_DENSITY_FRAC };
						} else {
							// Compute the local child offset of the grid coord
							const ei::UVec3 localCoord = gridCoord % 2u;
							const u32 localIdx = localCoord.x + 2u * (localCoord.y + 2u * localCoord.z);
							const u32 globalIdx = childIdx + localIdx;
							nextCellInfo[i] = { globalIdx, static_cast<float>(m_cellCounter[globalIdx]) };
							hadRefinement = true;
						}
					} else {
						// We already interpolated the current cell value, which means we no longer have
						// any information about the density distribution; thus we use the expected
						// distribution, which is quarter coverage
						nextCellInfo[i] = { 0u, currCellInfo[parentIdx].count * EXPECTED_DENSITY_FRAC };
					}
				}
				// Copy over the indices for the next run
				std::memcpy(currCellInfo, nextCellInfo, sizeof(currCellInfo));
				cellsPerDim <<= 1u;
			}

			// Interpolate final value (cell centers are the 0-1-borders)
			//         7         8         9
			// Lerp:  0.5  1|0  0.5  1|0  0.5
			//   0.5   ┌─────────┬─────────┐
			//         │         │         │
			//   1|0   │    0    │    1    │
			//         │         │ x       │
			//   0.5   ├─────────┼─────────┤
			//         │         │         │
			//   1|0   │    2    │    3    │
			//         │         │         │
			//   0.5   └─────────┴─────────┘
			// Gotta shift the cellsPerDim to the proper value (not the next-next dim)
			cellsPerDim >>= 2;
			const ei::Vec3 finalGridCoord = { normPos * cellsPerDim };
			// Round to the local grid center coordinate; with that we "snap" to the
			// center of the interpolation cube and are then guaranteed in range [0.5, -0.5]
			// (except at the bounding box edges, where we extrapolate
			const ei::IVec3 finalICoord = ei::clamp(ei::round(finalGridCoord), ei::IVec3{ 1 }, ei::IVec3{cellsPerDim - 1});
			const ei::Vec3 interpFactors = ei::clamp(finalGridCoord - finalICoord + 0.5f, 0.f, 1.f);
			// Trilinear interpolation in the cube gives us the density estimate
			const float densityZ0 = ei::bilerp(currCellInfo[0].count, currCellInfo[1].count, currCellInfo[2].count, currCellInfo[3].count,
											   interpFactors.x, interpFactors.y);
			const float densityZ1 = ei::bilerp(currCellInfo[4].count, currCellInfo[5].count, currCellInfo[6].count, currCellInfo[7].count,
											   interpFactors.x, interpFactors.y);
			const float interpDensity = ei::lerp(densityZ0, densityZ1, interpFactors.z);
			if(interpDensity > 0) {
				// To adjust for area we center a virtual cell around the position (instead of using a
				// grid aligned one, since we interpolate the values)
				const ei::Vec3 cellSize = 1.0f / (cellsPerDim * m_sceneSizeInv);
				const ei::Vec3 cellMin = offPos - cellSize / 2.f;
				const ei::Vec3 cellMax = cellMin + cellSize;
				float area = math::intersection_area(cellMin, cellMax, offPos, normal);
				if(size)
					*size = area;
				// We do not have the problem with too small an area that the uninterpolated
				// (and thus grid-aligned) virtual cell has since we're not in the cell corner
				return m_densityScale * interpDensity / area;
			}
			return 0.f;
		}

		float get_density(const ei::Vec3& pos, const ei::Vec3& normal, float* size = nullptr) {
			ei::Vec3 offPos = pos - m_minBound;
			ei::Vec3 normPos = offPos * m_sceneSizeInv;
			// Get the integer position on the finest level.
			u32 gridRes = 1 << m_depth.load();
			ei::UVec3 iPos{ normPos * gridRes };
			// Get root value. This will most certainly be a child pointer...
			u32 childIdx = m_cellChildIndices[0].load();
			u32 idx = 0;
			// The most significant bit in iPos distinguishes the children of the root node.
			// For each level, the next bit will be the relevant one.
			u32 currentLvlMask = gridRes;
			while(childIdx != 0) {
				currentLvlMask >>= 1;
				// Get the relative index of the child [0,7]
				idx = ((iPos.x & currentLvlMask) ? 1 : 0)
					+ ((iPos.y & currentLvlMask) ? 2 : 0)
					+ ((iPos.z & currentLvlMask) ? 4 : 0);
				// Add global offset
				idx += childIdx;
				childIdx = m_cellChildIndices[idx].load();
			}
			// Load the count
			const u32 count = m_cellCounter[idx];
			if(count > 0) {
				// Get the world space cell boundaries
				int currentGridRes = gridRes / currentLvlMask;
				ei::UVec3 cellPos = iPos / currentLvlMask;
				ei::Vec3 cellSize = 1.0f / (currentGridRes * m_sceneSizeInv);
				ei::Vec3 cellMin = cellPos * cellSize;
				ei::Vec3 cellMax = cellMin + cellSize;
				float area = math::intersection_area(cellMin, cellMax, offPos, normal);
				// Sometimes the above method returns zero. Therefore we restrict the
				// area to something larger then a hunderds part of an approximate cell area.
				float cellDiag = m_sceneScale / currentGridRes;
				float minArea = cellDiag * cellDiag;
				if(size) { *size = cellDiag; minArea *= 0.1f; } else minArea *= 0.01f;
				return m_densityScale * count / ei::max(minArea, area);
			}
			return 0.0f;
		}

		float get_density_robust(const ei::Vec3& pos, const scene::TangentSpace& ts) {
			float d[5];
			float cellDiag = 1e-3f;
			int count = 0;
			d[0] = get_density(pos, ts.geoN, &cellDiag);
			cellDiag *= 1.1f;
			if(d[0] > 0.0f) ++count;
			d[count] = get_density(pos + ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos - ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos + ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density(pos - ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			// Find the median via selection sort up to the element m.
			// Prefer the greater element, because overestimations do not
			// cause such visible artifacts.
			int m = count / 2;
			for(int i = 0; i <= m; ++i) for(int j = i + 1; j < count; ++j)
				if(d[j] < d[i])
					std::swap(d[i], d[j]);
			return d[m];
			/*float maxD = d[0];
			for(int i = 1; i < count; ++i)
				if(d[i] > maxD) maxD = d[i];
			return maxD;*/
		}

		float get_density_interpolated_robust(const ei::Vec3& pos, const scene::TangentSpace& ts) {
			float d[5];
			float cellDiag = 1e-3f;
			int count = 0;
			d[0] = get_density_interpolated(pos, ts.geoN, &cellDiag);
			cellDiag *= 1.1f;
			if(d[0] > 0.0f) ++count;
			d[count] = get_density_interpolated(pos + ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density_interpolated(pos - ts.shadingTX * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density_interpolated(pos + ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			d[count] = get_density_interpolated(pos - ts.shadingTY * cellDiag, ts.geoN);
			if(d[count] > 0.0f) ++count;
			// Find the median via selection sort up to the element m.
			// Prefer the greater element, because overestimations do not
			// cause such visible artifacts.
			int m = count / 2;
			for(int i = 0; i <= m; ++i) for(int j = i + 1; j < count; ++j)
				if(d[j] < d[i])
					std::swap(d[i], d[j]);
			return d[m];
			/*float maxD = d[0];
			for(int i = 1; i < count; ++i)
				if(d[i] > maxD) maxD = d[i];
			return maxD;*/
		}

		u32 capacity() const { return m_capacity; }
		u32 size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
		// Get the size of the associated memory excluding this instance.
		std::size_t mem_size() const { return sizeof(std::atomic_uint32_t) * m_capacity; }
	private:
		float m_densityScale;		// 1/#iterations to normalize the counters into a density
		u32 m_splitCountDensity;	// The number when a node is split must be a multiple of 8 and must grow proportional to #iterations
		ei::Vec3 m_minBound;
		ei::Vec3 m_sceneSizeInv;
		// Nodes consist of 8 atomic counters OR child indices. Each number is either a
		// counter (positive) or a negated child index.
		std::unique_ptr<std::atomic_uint32_t[]> m_cellCounter;
		std::unique_ptr<std::atomic_uint32_t[]> m_cellChildIndices;
		std::atomic_uint32_t m_allocationCounter;
		std::atomic_uint32_t m_depth;
		u32 m_capacity;
		float m_sceneScale;
		bool m_stopFilling;

		// Returns the new child pointer or 0
		u32 split_node_if_necessary(u32 idx, u32 childIdx, u32 count, u32 currentDepth) {
			// Only split leaf nodes
			// The node must be split if its density gets too high
			if(childIdx == 0 && count >= m_splitCountDensity && m_depth < MAX_TREE_DEPTH) {
				// Only one thread is responsible to do the allocation
				if(count == m_splitCountDensity) {
					u32 child = m_allocationCounter.fetch_add(8);
					if(child >= m_capacity) { // Allocation overflow
						m_allocationCounter.store(m_capacity + 1);	// Avoid overflow of the counter (but keep a large number)
						return 0;
					}
					// We do not know anything about the distribution of of photons -> equally
					// distribute. Therefore, all eight children are initilized with SPLIT_FACTOR on clear().
					m_cellChildIndices[idx].store(child);
					// Update depth
					atomic_max(m_depth, currentDepth + 1);
					// The current photon is already counted before the split -> return stop
					return 0;
				} else {
					// Spin-lock until the responsible thread has set the child pointer
					u32 nextChildIdx = m_cellChildIndices[idx].load();
					while(nextChildIdx == 0) {
						// Check for allocation overflow
						if(m_allocationCounter.load() > m_capacity)
							return 0;
						nextChildIdx = m_cellChildIndices[idx].load();
					}
					return nextChildIdx;
				}
			}
			return childIdx;
		}
	};

} // namespace mufflon::renderer