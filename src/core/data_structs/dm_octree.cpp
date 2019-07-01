#include "dm_octree.hpp"
#include "util/assert.hpp"
#include "util/types.hpp"

namespace mufflon::data_structs {

namespace {

template<typename T>
void atomic_max(std::atomic<T>& a, T b) {
	T oldV = a.load();
	while(oldV < b && !a.compare_exchange_weak(oldV, b));
}

} // namespace

DmOctree::DmOctree(const ei::Box& sceneBounds, const int capacity, const float splitFactor) :
	m_splitFactor{ [](const float factor) {
		if(factor <= 1.0f)
			logError("[DmOctree] Split factor must be larger than 1. Otherwise the tree will be split infinitely. Setting to 1.1 instead of ", factor);
		return ei::max(factor, 1.f);
	}(splitFactor) },
	m_sceneSize{ (sceneBounds.max - sceneBounds.min) * 2.002f },
	m_sceneSizeInv{ 1.f / m_sceneSize },
	m_minBound{ sceneBounds.min - m_sceneSize * (2.002f - 1.f) / 2.f }, // Slightly enlarge the volume to avoid numerical issues on the boundary
	m_capacity{ 1 + ((capacity + 7) & (~7)) },
	m_nodes{ std::make_unique<std::atomic_int32_t[]>(m_capacity) },
	m_allocationCounter{ 1 },
	m_depth{ 0 },
	m_stopFilling{ false }
{
	m_nodes[0].store(0);
	// TODO: parallelize?
	// The other nodes are only used if the parent is split
	//for(int i = 1; i < m_capacity; ++i)
	//	m_nodes[i].store(ei::ceil(SPLIT_FACTOR));
}

DmOctree::DmOctree(DmOctree&& octree) :
	m_densityScale(octree.m_densityScale),
	m_splitFactor(octree.m_splitFactor),
	m_splitCountDensity(octree.m_splitCountDensity),
	m_sceneSize(octree.m_sceneSize),
	m_sceneSizeInv(octree.m_sceneSizeInv),
	m_minBound(octree.m_minBound),
	m_capacity(octree.m_capacity),
	m_nodes(std::move(octree.m_nodes)),
	m_allocationCounter(octree.m_allocationCounter.load()),
	m_depth(octree.m_depth.load()),
	m_stopFilling(octree.m_stopFilling)
{}

void DmOctree::set_iteration(const int iter) {
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

void DmOctree::clear_counters() {
	int n = m_allocationCounter.load();
	for(int i = 0; i < n; ++i)
		if(m_nodes[i].load() > 0)
			m_nodes[i].store(0);
}

void DmOctree::clear() {
	m_allocationCounter.store(1);
	m_nodes[0].store(0);
	m_depth.store(0);
}

void DmOctree::increase_count(const ei::Vec3& pos) {
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

float DmOctree::get_density(const ei::Vec3& pos, const ei::Vec3& normal) const {
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	// Get the integer position on the finest level.
	int gridRes = 1 << m_depth.load();
	ei::IVec3 iPos{ normPos * gridRes };
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

template < bool UseSmoothStep >
float DmOctree::get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const {
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	// Get the integer position on the finest level.
	int maxLvl = m_depth.load();
	int gridRes = 1 << (maxLvl + 1);
	ei::IVec3 iPos{ normPos * gridRes };
	// Memory to track nodes
	int buffer[16];
	float areaBuffer[16];
	int* parents = buffer;
	int* current = buffer + 8;
	float* parentArea = areaBuffer;
	float* currentArea = areaBuffer + 8;
	for(int i = 0; i < 8; ++i) {
		current[i] = 0;	// Initialize to root level
		currentArea[i] = -1;
	}
	int lvl = 0;
	ei::IVec3 parentMinPos{ 0 };
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
		for(int i = 0; i < 8; ++i) {
			const int c = m_nodes[current[i]].load();
			anyHadChildren |= c < 0;
			// Density not yet computed? Density might have been copied
			// from parent in which case we do not need to compute it anew.
			if(c >= 0 && currentArea[i] < 0.0f) {
				const int ix = i & 1, iy = (i >> 1) & 1, iz = i >> 2;
				const ei::Vec3 localPos = offPos - (lvlPos + ei::IVec3{ ix, iy, iz }) * cellSize;
				const float area = math::intersection_area_nrm(cellSize, localPos, normal);
				currentArea[i] = area;
			}
		}
	}
	// The loop terminates if all 8 cells in 'current' are leaves.
	// This means we want to interpolate 'current' on 'lvl'.
	const int lvlRes = 1 << lvl;
	ei::Vec3 tPos = normPos * lvlRes - 0.5f;
	ei::IVec3 gridPos = ei::floor(tPos);
	ei::Vec3 ws[2];
	ws[1] = tPos - gridPos;
	ws[0] = 1.0f - ws[1];

	float countSum = 0.0f, areaSum = 0.0f;
	const ei::Vec3 cellSize{ m_sceneSize / lvlRes };
	const float avgArea = (cellSize.x * cellSize.y + cellSize.x * cellSize.z + cellSize.y * cellSize.z) / 3.0f;
	for(int i = 0; i < 8; ++i) {
		const int ix = i & 1, iy = (i >> 1) & 1, iz = i >> 2;
		const ei::Vec3 localPos = offPos - (gridPos + ei::IVec3{ ix, iy, iz }) * cellSize;
		const float area = math::intersection_area_nrm(cellSize, localPos, normal);
		// Compute trilinear interpolated result of count and area (running sum)
		const float w = UseSmoothStep ? ei::smoothstep(ws[ix].x) * ei::smoothstep(ws[iy].y) * ei::smoothstep(ws[iz].z)
									  : ws[ix].x * ws[iy].y * ws[iz].z;
		mAssert(m_nodes[current[i]].load() >= 0);
		if(area > 0.0f && currentArea[i] > 0.0f) {
			float lvlFactor = (area + avgArea * 0.01f) / (currentArea[i] + avgArea * 0.01f);
			const float count = static_cast<float>(m_nodes[current[i]].load());
			const float weightedCount = count * w * lvlFactor;
			const float weightedArea = area * w;

			if(gradient != nullptr) {
				if constexpr(UseSmoothStep) {
					// Derivative for smooth step
					*gradient += ei::Vec3{
						(ix ? -1.f : 1.f) * 6.f * ws[1].x * ws[0].x * ws[iy].y * ws[iz].z * count,
						(iy ? -1.f : 1.f) * 6.f * ws[1].y * ws[0].y * ws[iy].x * ws[iz].z * count,
						(iz ? -1.f : 1.f) * 6.f * ws[1].z * ws[0].z * ws[iy].x * ws[iz].y * count,
					};
				} else {
					// Gradient for trilinear interpolation
					*gradient += ei::Vec3{
						(ix ? -1.f : 1.f) * ws[iy].y * ws[iz].z * count,
						(iy ? -1.f : 1.f) * ws[ix].x * ws[iz].z * count,
						(iz ? -1.f : 1.f) * ws[ix].x * ws[iy].y * count
					};
				}
			}
			countSum += weightedCount;
			areaSum += weightedArea;
		}
	}

	if(gradient != nullptr) {
		*gradient = sdiv(*gradient, areaSum);
	}
	mAssert(areaSum > 0.0f);
	return sdiv(countSum, areaSum) * m_densityScale;
}

void DmOctree::balance(const int current, const int nx, const int ny, const int nz,
					   const int px, const int py, const int pz) {
	// Call balance for each child recursively, if the child has children itself.
	// Otherwise balance is satisfied.
	int children = -m_nodes[current].load();
	if(children <= 0) return;	// No tree here
	for(int i = 0; i < 8; ++i) {
		int childC = m_nodes[children + i].load();
		if(childC < 0) {
			// To make the recursive call we need all the neighbors on the child-level.
			// If they do not extist we need to split the respective cell.
			const ei::IVec3 localPos{ i & 1, (i >> 1) & 1, i >> 2 };
			const int cnx = find_neighbor(localPos, 0, 0, nx, children);
			const int cny = find_neighbor(localPos, 1, 0, ny, children);
			const int cnz = find_neighbor(localPos, 2, 0, nz, children);
			const int cpx = find_neighbor(localPos, 0, 1, px, children);
			const int cpy = find_neighbor(localPos, 1, 1, py, children);
			const int cpz = find_neighbor(localPos, 2, 1, pz, children);
			balance(children + i, cnx, cny, cnz, cpx, cpy, cpz);
		}
	}
}


int DmOctree::find_neighbor(const ei::IVec3& localPos, const int dim, const int dir,
							const int parentNeighbor, const int siblings) {
	int cn = 0;	// Initialize to outer boundary
	// The adoint position is the child index of the neighbor (indepndent of the parent).
	// It merely flips the one coordinate of the relevant dimension
	int adjointIdx = 0;
	adjointIdx += (dim == 0 ? 1 - localPos[0] : localPos[0]);
	adjointIdx += (dim == 1 ? 1 - localPos[1] : localPos[1]) * 2;
	adjointIdx += (dim == 2 ? 1 - localPos[2] : localPos[2]) * 4;
	if(localPos[dim] == dir && parentNeighbor > 0) { // Not on boundary
		int nC = m_nodes[parentNeighbor].load();
		if(nC >= 0) nC = split(parentNeighbor);
		cn = -nC + adjointIdx;
	} else if(localPos[dim] == (1 - dir)) {
		cn = siblings + adjointIdx;
	}
	return cn;
}
int DmOctree::increment_if_positive(const int idx) {
	int oldV = m_nodes[idx].load();
	int newV;
	do {
		if(oldV < 0) return oldV;	// Do nothing, the value is a child pointer
		newV = oldV + 1;			// Increment
	} while(!m_nodes[idx].compare_exchange_weak(oldV, newV));	// Write if nobody destroyed the value
	return newV;
}

int DmOctree::split_node_if_necessary(const int idx, const int count, const int currentDepth) {
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
			atomic_max(m_depth, currentDepth + 1);
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

int DmOctree::split(const int idx) {
	int child = m_allocationCounter.fetch_add(8);
	if(child >= m_capacity) { // Allocation overflow
		m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
		return 0;
	}
	m_nodes[idx].store(-child);
	return -child;
}

template float DmOctree::get_density_interpolated<true>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;
template float DmOctree::get_density_interpolated<false>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;

} // namespace mufflon::data_structs