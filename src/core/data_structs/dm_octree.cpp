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

template < class T >
DmOctree<T>::DmOctree(const ei::Box& sceneBounds, const int capacity, const float splitFactor,
					  const float progression) :
	m_splitFactor{ [](const float factor) {
		if(factor <= 1.0f)
			logError("[DmOctree] Split factor must be larger than 1. Otherwise the tree will be split infinitely. Setting to 1.1 instead of ", factor);
		return ei::max(factor, 1.f);
	}(splitFactor) },
	m_sceneSize{ (sceneBounds.max - sceneBounds.min) * 2.002f },
	m_sceneSizeInv{ 1.f / m_sceneSize },
	m_minBound{ sceneBounds.min - m_sceneSize * (2.002f - 1.f) / 2.f }, // Slightly enlarge the volume to avoid numerical issues on the boundary
	m_capacity{ 1 + ((capacity + 7) & (~7)) },
	m_progression{ progression },
	m_nodes{ std::make_unique<std::atomic<T>[]>(m_capacity) },
	m_allocationCounter{ 1 },
	m_depth{ 0 },
	m_stopFilling{ false }
{
	m_nodes[0].store(T{ 0 });
	clear();
	// TODO: parallelize?
	// The other nodes are only used if the parent is split
	//for(int i = 1; i < m_capacity; ++i)
	//	m_nodes[i].store(ei::ceil(SPLIT_FACTOR));
}

template < class T >
DmOctree<T>::DmOctree(DmOctree&& octree) :
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
	m_progression(octree.m_progression),
	m_stopFilling(octree.m_stopFilling)
{}

template < class T >
void DmOctree<T>::set_iteration(const int iter) {
	int iterClamp = ei::min(FILL_ITERATIONS, iter);
	m_stopFilling = iter > FILL_ITERATIONS;
	m_densityScale = 1.0f / iterClamp;
	m_splitCountDensity = static_cast<T>(ei::pow(static_cast<T>(iterClamp), m_progression));
}

template < class T >
void DmOctree<T>::clear_counters() {
	int n = m_allocationCounter.load();
	for(int i = 0; i < n; ++i)
		if(m_nodes[i].load() > T{ 0 })
			m_nodes[i].store(T{ 0 });
}

template < class T >
void DmOctree<T>::clear() {
	m_allocationCounter.store(1);
	m_depth.store(3u);
	// Split the first 3 levels
	constexpr int NUM_SPLIT_NODES = 1 + 8 + 64;
	constexpr int NUM_LEAVES = 512;
	for(int i = 0; i < NUM_SPLIT_NODES; ++i)
		split(i);
	// Set counters in leaf nodes to 0
	for(int i = 0; i < NUM_LEAVES; ++i)
		m_nodes[NUM_SPLIT_NODES + i].store(0);
}

template < class T >
void DmOctree<T>::increase_count(const ei::Vec3& pos, const ei::Vec3& normal, const T value) {
	// TODO: proper value
	if(m_stopFilling) return;
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	ei::IVec3 iPos{ normPos * (1 << 30) };
	T countOrChild = mark_child_pointer(1);
	int lvl = 1;
	do {
		// Get the relative index of the child [0, 7]
		const ei::IVec3 gridPos = iPos >> (30 - lvl);
		int idx = (gridPos.x & 1) + 2 * (gridPos.y & 1) + 4 * (gridPos.z & 1);
		idx -= static_cast<int>(countOrChild);	// 'Add' global offset (which is stored negative)
		countOrChild = increment_if_child_and_split_if_necessary(idx, value, lvl, gridPos, offPos, normal);
		++lvl;
	} while(countOrChild < 0);
}

template < class T >
template < class V >
V DmOctree<T>::get_density(const ei::Vec3& pos, const ei::Vec3& normal) const {
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	// Get the integer position on the finest level.
	int gridRes = 1 << m_depth.load();
	ei::IVec3 iPos{ normPos * gridRes };
	// Get root value. This will most certainly be a child pointer...
	T countOrChild = m_nodes[0].load();
	// The most significant bit in iPos distinguishes the children of the root node.
	// For each level, the next bit will be the relevant one.
	int currentLvlMask = gridRes;
	while(is_child_pointer(countOrChild)) {
		currentLvlMask >>= 1;
		// Get the relative index of the child [0,7]
		int idx = ((iPos.x & currentLvlMask) ? 1 : 0)
			+ ((iPos.y & currentLvlMask) ? 2 : 0)
			+ ((iPos.z & currentLvlMask) ? 4 : 0);
		// 'Add' global offset (which is stored negative)
		idx -= static_cast<int>(countOrChild);
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
	return T{ 0 };
}

template < class T >
template < bool UseSmoothStep, class V >
V DmOctree<T>::get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const {
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
	bool anyHadChildren = is_child_pointer(m_nodes[0].load());
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
		const ei::Vec3 cellSize = m_sceneSize / lvlRes;
		anyHadChildren = false; // Check for the children inside the for loop
		for(int i = 0; i < 8; ++i) {
			ei::IVec3 cellPos = lvlPos + CELL_ITER[i];
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
			T c = m_nodes[parentAddress].load();
			if(is_child_pointer(c)) {
				// Insert the child node's address
				int localChildIdx = (cellPos.x & 1) + 2 * (cellPos.y & 1) + 4 * (cellPos.z & 1);
				current[i] = static_cast<int>(-c) + localChildIdx;
				//currentArea[i] = -1.0f;
				const T cc = m_nodes[current[i]].load();
				anyHadChildren |= is_child_pointer(cc);
				// Compute the area if this is a leaf node
				if(!is_child_pointer(cc)) {
					const ei::Vec3 localPos = offPos - cellPos * cellSize;
					const float area = math::intersection_area_nrm(cellSize, localPos, normal);
					currentArea[i] = -area;	// Encode that this is new
				}
			} else { // Otherwise copy the parent to the next finer level.
				current[i] = parentAddress;
				currentArea[i] = ei::abs(parentArea[parentIdx]);
			}
		}
		parentMinPos = lvlPos * 2;
	}
	// The loop terminates if all 8 cells in 'current' are leaves.
	// This means we want to interpolate 'current' on 'lvl'.
	const int lvlRes = 1 << lvl;
	ei::Vec3 tPos = normPos * lvlRes - 0.5f;
	ei::IVec3 gridPos = ei::floor(tPos);
	ei::Vec3 ws[2];
	ws[1] = tPos - gridPos;
	ws[0] = 1.0f - ws[1];

	V countSum = 0.0f, areaSum = 0.0f;
	const ei::Vec3 cellSize{ m_sceneSize / lvlRes };
	const float eps = (0.01f / 3.0f) * (cellSize.x * cellSize.y + cellSize.x * cellSize.z + cellSize.y * cellSize.z);
	for(int i = 0; i < 8; ++i) {
		const ei::Vec3 localPos = offPos - (gridPos + CELL_ITER[i]) * cellSize;
		float lvlFactor = 1.f;
		float area;
		if(currentArea[i] > 0.f) {
			area = math::intersection_area_nrm(cellSize, localPos, normal);
			lvlFactor = (area + eps) / (currentArea[i] + eps);
		} else {
			area = ei::abs(currentArea[i]);
		}
		// Compute trilinear interpolated result of count and area (running sum)
		mAssert(m_nodes[current[i]].load() >= 0);
		const float w = UseSmoothStep ? ei::smoothstep(ws[CELL_ITER[i].x].x) * ei::smoothstep(ws[CELL_ITER[i].y].y) * ei::smoothstep(ws[CELL_ITER[i].z].z)
			: ws[CELL_ITER[i].x].x * ws[CELL_ITER[i].y].y * ws[CELL_ITER[i].z].z;
		if(area > 0.0f) {
			const V count = static_cast<V>(m_nodes[current[i]].load());
			const V weightedCount = count * w * lvlFactor;
			const float weightedArea = area * w;

			if(gradient != nullptr) {
				if constexpr(UseSmoothStep) {
					// Derivative for smooth step
					// TODO: are those indices correct?
					*gradient += ei::Vec3{
						(CELL_ITER[i].x ? -1.f : 1.f) * 6.f * ws[1].x * ws[0].x * ws[CELL_ITER[i].y].y * ws[CELL_ITER[i].z].z * count,
						(CELL_ITER[i].y ? -1.f : 1.f) * 6.f * ws[1].y * ws[0].y * ws[CELL_ITER[i].y].x * ws[CELL_ITER[i].z].z * count,
						(CELL_ITER[i].z ? -1.f : 1.f) * 6.f * ws[1].z * ws[0].z * ws[CELL_ITER[i].y].x * ws[CELL_ITER[i].z].y * count,
					};
				} else {
					// Gradient for trilinear interpolation
					*gradient += ei::Vec3{
						(CELL_ITER[i].x ? -1.f : 1.f) * ws[CELL_ITER[i].y].y * ws[CELL_ITER[i].z].z * count,
						(CELL_ITER[i].y ? -1.f : 1.f) * ws[CELL_ITER[i].x].x * ws[CELL_ITER[i].z].z * count,
						(CELL_ITER[i].z ? -1.f : 1.f) * ws[CELL_ITER[i].x].x * ws[CELL_ITER[i].y].y * count
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

template < class T >
void DmOctree<T>::balance(const int current, const int nx, const int ny, const int nz,
						  const int px, const int py, const int pz) {
	// Call balance for each child recursively, if the child has children itself.
	// Otherwise balance is satisfied.
	int children = static_cast<int>(-m_nodes[current].load());
	if(children <= 0) return;	// No tree here
	for(int i = 0; i < 8; ++i) {
		T childC = m_nodes[children + i].load();
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

template < class T >
int DmOctree<T>::find_neighbor(const ei::IVec3& localPos, const int dim, const int dir,
							const int parentNeighbor, const int siblings) {
	int cn = 0;	// Initialize to outer boundary
	// The adoint position is the child index of the neighbor (indepndent of the parent).
	// It merely flips the one coordinate of the relevant dimension
	int adjointIdx = 0;
	adjointIdx += (dim == 0 ? 1 - localPos[0] : localPos[0]);
	adjointIdx += (dim == 1 ? 1 - localPos[1] : localPos[1]) * 2;
	adjointIdx += (dim == 2 ? 1 - localPos[2] : localPos[2]) * 4;
	if(localPos[dim] == dir && parentNeighbor > 0) { // Not on boundary
		T nC = m_nodes[parentNeighbor].load();
		if(!is_child_pointer(nC)) nC = split(parentNeighbor);
		cn = static_cast<int>(-nC) + adjointIdx;
	} else if(localPos[dim] == (1 - dir)) {
		cn = siblings + adjointIdx;
	}
	return cn;
}

template < class T >
T DmOctree<T>::increment_if_child_and_split_if_necessary(const int idx, const T value, const int currentDepth,
														 const ei::IVec3& gridPos, const ei::Vec3& offPos,
														 const ei::Vec3& normal) {
	// Too large depths would break the integer arithmetic in the grid.
	if(currentDepth >= 30)
		return T(0);
	T oldV = m_nodes[idx].load();
	T newV;
	do {
		if(is_child_pointer(oldV)) return oldV;	// Do nothing, the value is a child pointer
		newV = oldV + value;			// Increment
	} while(!m_nodes[idx].compare_exchange_weak(oldV, newV));	// Write if nobody destroyed the value

	// TODO: buggy
	
	// The node must be split if its density gets too high
	if(newV >= m_splitCountDensity) {
		// Only one thread is responsible to do the allocation
		// We also only perform one split at a time
		if(newV >= m_splitCountDensity && oldV < m_splitCountDensity) {
			const int child = m_allocationCounter.fetch_add(8);
			if(child >= m_capacity) { // Allocation overflow
				m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
				return 0;
			}
			init_children(child, newV, currentDepth, gridPos, offPos, normal);
			//init_children(child, newV);
			// We do not know anything about the distribution of of photons -> equally
			// distribute. Therefore, all eight children are initilized with SPLIT_FACTOR on clear().
			m_nodes[idx].store(mark_child_pointer(static_cast<T>(child)));
			// Update depth
			atomic_max(m_depth, currentDepth + 1);
			// The current photon is already counted before the split -> return stop
			return 0;
		} else {
			// Spin-lock until the responsible thread has set the child pointer
			T child = m_nodes[idx].load();
			while(child > 0) {
				// Check for allocation overflow
				if(m_allocationCounter.load() > m_capacity)
					return 0;
				child = m_nodes[idx].load();
			}
			return child;
		}
	}
	return newV;
}

template < class T >
T DmOctree<T>::split(const int idx) {
	int child = m_allocationCounter.fetch_add(8);
	if(child >= m_capacity) { // Allocation overflow
		m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
		return 0;
	}
	m_nodes[idx].store(mark_child_pointer(static_cast<T>(child)));
	return mark_child_pointer(static_cast<T>(child));
}

// Set the counter of all unused cells to the number of expected samples
// times 2. A planar surface will never extend to all eight cells. It might
// intersect 7 of them, but still the distribution is one of a surface.
// Therefore, the factor 2 (distribute among 4 cells) gives a much better initial
// value.
template < class T >
void DmOctree<T>::init_children(const int children, const T count) {
	for(int i = 0; i < 8; ++i)
		m_nodes[children + i].store(static_cast<T>(ei::ceil(m_splitCountDensity / 4.0f)));
}

template < class T >
void DmOctree<T>::init_children(const int children, const T count, const int currentDepth,
								const ei::IVec3& gridPos, const ei::Vec3& offPos,
								const ei::Vec3& normal) {
	ei::Vec3 childCellSize = m_sceneSize / (1 << (currentDepth + 1));
	ei::Vec3 localPos = offPos - gridPos * 2 * childCellSize;
	// Get the intersection areas of the eight children to distribute
	// the count properly.
	float area[8];
	float areaSum = 0.0f;
	for(int i = 0; i < 8; ++i) {
		const ei::IVec3 childLocalPos{ i & 1, (i >> 1) & 1, i >> 2 };
		area[i] = math::intersection_area_nrm(childCellSize, localPos - childLocalPos * childCellSize, normal);
		//area[i] = math::intersection_area_nrm(childCellSize * 1.5f, localPos - (childLocalPos - 0.25f) * childCellSize, normal);
		areaSum += area[i];
	}
	const T minCount = static_cast<T>(ei::ceil(count / 8.0f));
	// Distribute the count proportional to the areas. To avoid loss we cannot
	// simply round. https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
	float cumVal = 0.0f;
	int prevCumRounded = 0;
	for(int i = 0; i < 8; ++i) {
		cumVal += area[i] / areaSum * count;
		const int cumRounded = ei::round(cumVal);
		// The min(count-1) is necessary to avoid a child cell which itself
		// already has the split count -> would lead to a dead lock.
		//int subCount = ei::min(count - 1, cumRounded - prevCumRounded); // More correct
		const T subCount = ei::clamp(static_cast<T>(cumRounded - prevCumRounded), minCount, count - 1);
		//int subCount = minCount;
		m_nodes[children + i].store(subCount);
		prevCumRounded = cumRounded;
	}
}

template class DmOctree<i32>;
template float DmOctree<i32>::get_density<float>(const ei::Vec3& pos, const ei::Vec3& normal) const;
template float DmOctree<i32>::get_density_interpolated<true, float>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;
template float DmOctree<i32>::get_density_interpolated<false, float>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;
template class DmOctree<float>;
template float DmOctree<float>::get_density<float>(const ei::Vec3& pos, const ei::Vec3& normal) const;
template float DmOctree<float>::get_density_interpolated<true, float>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;
template float DmOctree<float>::get_density_interpolated<false, float>(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient) const;

} // namespace mufflon::data_structs