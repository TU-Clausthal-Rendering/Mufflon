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
DmOctree<T>::DmOctree(const ei::Box& sceneBounds, int capacity, float splitFactor, bool progressive) :
	m_densityScale{},
	m_splitFactor{ ei::max(1.1f, splitFactor) },
	m_splitCountDensity{},
	m_progression{ progressive ? 1.f : 0.f },
	m_sceneSize{ (sceneBounds.max - sceneBounds.min) * 2.002f },
	m_sceneSizeInv{ 1.f / m_sceneSize },
	m_minBound{ sceneBounds.min - m_sceneSize * (2.002f - 1.f) / 2.f },
	m_capacity{ 1 + ((capacity + 7) & (~7)) },
	m_nodes{ std::make_unique<NodeData[]>(m_capacity) },
	m_allocationCounter{},
	m_depth{},
	m_stopFilling{ false }
{
	if(splitFactor <= 1.0f)
		logError("[DmOctree] Split factor must be larger than 1. Otherwise the tree will be split infinitely. Setting to 1.1 instead of ", splitFactor);
	clear();
}

template < class T >
DmOctree<T>::DmOctree(DmOctree&& other) :
	m_densityScale{ other.m_densityScale },
	m_splitFactor{ other.m_splitFactor },
	m_splitCountDensity{ other.m_splitCountDensity },
	m_progression{ other.m_progression },
	m_sceneSize{ other.m_sceneSize },
	m_sceneSizeInv{ other.m_sceneSizeInv },
	m_minBound{ other.m_minBound },
	m_capacity{ other.m_capacity },
	m_nodes{ std::move(other.m_nodes) },
	m_allocationCounter{ other.m_allocationCounter.load() },
	m_depth{ other.m_depth.load() },
	m_stopFilling{ other.m_stopFilling }
{}

template < class T >
void DmOctree<T>::clear() {
	m_allocationCounter.store(1);
	m_depth.store(3);
	// Split the first 3 levels
	constexpr int NUM_SPLIT_NODES = 1 + 8 + 64;
	constexpr int NUM_LEAVES = 512;
	for(int i = 0; i < NUM_SPLIT_NODES; ++i)
		split(i);
	// Set counters in leaf nodes to 0
	for(int i = 0; i < NUM_LEAVES; ++i)
		m_nodes[NUM_SPLIT_NODES + i] = DataType{};
}

template < class T >
void DmOctree<T>::set_iteration(int iter) {
	int iterClamp = ei::min(FILL_ITERATIONS, iter);
	m_stopFilling = iter > FILL_ITERATIONS;
	m_densityScale = 1.0f / iterClamp;
	m_splitCountDensity = ei::ceil(m_splitFactor * powf(float(iterClamp), m_progression));
}

template < class T >
void DmOctree<T>::clear_counters() {
	int n = m_allocationCounter.load();
	for(int i = 0; i < n; ++i)
		if(!has_children(m_nodes[i]))
			m_nodes[i] = DataType{};
}

template < class T >
void DmOctree<T>::increase_count(const ei::Vec3& pos, const ei::Vec3& normal, const DataType& value) {
	if(m_stopFilling) return;
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	ei::IVec3 iPos{ normPos * (1 << 30) };
	DataType countOrChild = -1;
	int lvl = 1;
	do {
		// Get the relative index of the child [0,7]
		ei::IVec3 gridPos = iPos >> (30 - lvl);
		int idx = (gridPos.x & 1) + 2 * (gridPos.y & 1) + 4 * (gridPos.z & 1);
		idx += get_child_offset(countOrChild);	// 'Add' global offset (which is stored negative)

		{
			//std::scoped_lock lock{ m_nodes[idx].lock };
			countOrChild = increment_if_positive(idx, value);
			countOrChild = split_node_if_necessary(idx, countOrChild, lvl, gridPos, offPos, normal);
		}
		++lvl;
	} while(has_children(countOrChild));
}

template < class T >
float DmOctree<T>::get_density(const ei::Vec3& pos, const ei::Vec3& normal) const {
	ei::Vec3 offPos = pos - m_minBound;
	ei::Vec3 normPos = offPos * m_sceneSizeInv;
	// Get the integer position on the finest level.
	int gridRes = 1 << m_depth.load();
	ei::IVec3 iPos{ normPos * gridRes };
	// Get root value. This will most certainly be a child pointer...
	DataType countOrChild = m_nodes[0];
	// The most significant bit in iPos distinguishes the children of the root node.
	// For each level, the next bit will be the relevant one.
	int currentLvlMask = gridRes;
	while(has_children(countOrChild)) {
		currentLvlMask >>= 1;
		// Get the relative index of the child [0,7]
		int idx = ((iPos.x & currentLvlMask) ? 1 : 0)
			+ ((iPos.y & currentLvlMask) ? 2 : 0)
			+ ((iPos.z & currentLvlMask) ? 4 : 0);
		// 'Add' global offset (which is stored negative)
		idx += get_child_offset(countOrChild);
		countOrChild = m_nodes[idx];
	}
	if(countOrChild > 0) {
		// Get the world space cell boundaries
		currentLvlMask = ei::max(1, currentLvlMask);
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

template < class T >
float DmOctree<T>::get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal) const {
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
		currentArea[i] = 0.0f;
	}
	int lvl = 0;
	ei::IVec3 parentMinPos{ 0 };
	bool anyHadChildren = has_children(m_nodes[0]);
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
		anyHadChildren = false;	// Check for the new children inside the for loop
		for(int i = 0; i < 8; ++i) {
			ei::IVec3 cellPos = lvlPos + CELL_ITER[i];
			// We need to find the parent in the 'parents' buffer array.
			// Since the window of interpolation moves the reference coordinate
			// we subtract 'parentMinPos' scaled to the current level.
			ei::IVec3 localParent = (cellPos - parentMinPos) / 2;
			mAssert(localParent >= 0 && localParent <= 1);
			//ei::IVec3 localParent = 1 - (clamp(cellPos, 1, lvlRes-2) & 1);
			//ei::IVec3 localParent = 1 - (cellPos & 1);
			int parentIdx = localParent.x + 2 * localParent.y + 4 * localParent.z;
			// Check if parent node has children.
			int parentAddress = parents[parentIdx];
			DataType c = m_nodes[parentAddress];
			if(has_children(c)) {
				// Insert the child node's address
				int localChildIdx = (cellPos.x & 1) + 2 * (cellPos.y & 1) + 4 * (cellPos.z & 1);
				current[i] = get_child_offset(c) + localChildIdx;
				//currentArea[i] = -1.0f;
				const DataType cc = m_nodes[current[i]];
				anyHadChildren |= has_children(cc);
				// Compute the area if this is a leaf node
				if(!has_children(cc)) {
					const ei::Vec3 localPos = offPos - cellPos * cellSize;
					const float area = math::intersection_area_nrm(cellSize, localPos, normal);
					currentArea[i] = -area; // Encode that this is new
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
	float countSum = 0.0f, areaSum = 0.0f;
	const ei::Vec3 cellSize{ m_sceneSize / lvlRes };
	const float eps = (0.01f / 3.0f) * (cellSize.x * cellSize.y + cellSize.x * cellSize.z + cellSize.y * cellSize.z);
	for(int i = 0; i < 8; ++i) {
		const ei::Vec3 localPos = offPos - (gridPos + CELL_ITER[i]) * cellSize;
		float lvlFactor = 1.0f;
		float area;
		if(currentArea[i] > 0.0f) { // Only needs compensation if not on the same level
			area = math::intersection_area_nrm(cellSize, localPos, normal);
			lvlFactor = (area + eps) / (currentArea[i] + eps);
		} else area = ei::abs(currentArea[i]);
		// Compute trilinear interpolated result of count and area (running sum)
		mAssert(!has_children(m_nodes[current[i]]));
		if(area > 0.0f) {
			const float w = ws[CELL_ITER[i].x].x * ws[CELL_ITER[i].y].y * ws[CELL_ITER[i].z].z;
			countSum += m_nodes[current[i]] * w * lvlFactor;
			areaSum += area * w;
		}
	}
	mAssert(areaSum > 0.0f);
	return sdiv(countSum, areaSum) * m_densityScale;
}

template < class T >
void DmOctree<T>::balance(int current, int nx, int ny, int nz, int px, int py, int pz) {
	// Call balance for each child recursively, if the child has children itself.
	// Otherwise balance is satisfied.
	int children = get_child_offset(m_nodes[current]);
	if(children <= 0) return;	// No tree here
	for(int i = 0; i < 8; ++i) {
		const DataType childC = m_nodes[children + i];
		if(has_children(childC)) {
			// To make the recursive call we need all the neighbors on the child-level.
			// If they do not extist we need to split the respective cell.
			const ei::IVec3 localPos{ i & 1, (i >> 1) & 1, i >> 2 };
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

template < class T >
int DmOctree<T>::find_neighbor(const ei::IVec3& localPos, int dim, int dir, int parentNeighbor, int siblings) {
	int cn = 0;	// Initialize to outer boundary
	// The adoint position is the child index of the neighbor (indepndent of the parent).
	// It merely flips the one coordinate of the relevant dimension
	int adjointIdx = 0;
	adjointIdx += (dim == 0 ? 1 - localPos[0] : localPos[0]);
	adjointIdx += (dim == 1 ? 1 - localPos[1] : localPos[1]) * 2;
	adjointIdx += (dim == 2 ? 1 - localPos[2] : localPos[2]) * 4;
	if(localPos[dim] == dir && parentNeighbor > 0) { // Not on boundary
		DataType nC = m_nodes[parentNeighbor];
		if(!has_children(nC)) nC = split(parentNeighbor);
		cn = get_child_offset(nC) + adjointIdx;
	} else if(localPos[dim] == (1 - dir)) {
		cn = siblings + adjointIdx;
	}
	return cn;
}

template < class T >
bool DmOctree<T>::has_children(const DataType& val) const noexcept {
	return val < 0;
}

template < class T >
int DmOctree<T>::get_child_offset(const DataType& val) const noexcept {
	return -static_cast<int>(val);
}

template < class T >
typename DmOctree<T>::DataType DmOctree<T>::create_child_offset(const int offset) const noexcept {
	return static_cast<DataType>(-offset);
}

template < class T >
typename DmOctree<T>::DataType DmOctree<T>::increment_if_positive(int idx, const DataType& value) {
	DataType oldV = m_nodes[idx];
	DataType newV;
	do {
		if(has_children(oldV)) return oldV;	// Do nothing, the value is a child pointer
		newV = oldV + value;			// Increment
	} while(!(*m_nodes[idx]).compare_exchange_weak(oldV, newV));	// Write if nobody destroyed the value
	return newV;
}

template < class T >
typename DmOctree<T>::DataType DmOctree<T>::split_node_if_necessary(int idx, DataType value, int currentDepth,
																	const ei::IVec3& gridPos, const ei::Vec3& offPos,
																	const ei::Vec3& normal) {
	// Too large depths would break the integer arithmetic in the grid.
	if(currentDepth >= 30) return 0;
	// The node must be split if its density gets too high
	if(value >= m_splitCountDensity) {
		int child = m_allocationCounter.fetch_add(8);
		if(child >= m_capacity) { // Allocation overflow
			m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
			return 0;
		}
		init_children(child, value, currentDepth, gridPos, offPos, normal);
		//init_children(child, count);
		// We do not know anything about the distribution of of photons -> equally
		// distribute. Therefore, all eight children are initilized with SPLIT_FACTOR on clear().
		m_nodes[idx] = create_child_offset(child);
		// Update depth
		atomic_max(m_depth, currentDepth + 1);
		// The current photon is already counted before the split -> return stop
		return 0;
	}
	return value;
}

template < class T >
typename DmOctree<T>::DataType DmOctree<T>::split(int idx) {
	int child = m_allocationCounter.fetch_add(8);
	if(child >= m_capacity) { // Allocation overflow
		m_allocationCounter.store(int(m_capacity + 1));	// Avoid overflow of the counter (but keep a large number)
		return 0;
	}
	const auto offset = create_child_offset(child);
	m_nodes[idx] = offset;
	return offset;
}

template < class T >
void DmOctree<T>::init_children(int children, DataType /*count*/) {
	for(int i = 0; i < 8; ++i)
		m_nodes[children + i] = static_cast<DataType>(ei::ceil(m_splitCountDensity / 4.0f));
}

template < class T >
void DmOctree<T>::init_children(int children, DataType count, int currentDepth,
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
	int minCount = ei::ceil(count / 8.0f);
	// Distribute the count proportional to the areas. To avoid loss we cannot
	// simply round. https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
	float cumVal = 0.0f;
	int prevCumRounded = 0;
	for(int i = 0; i < 8; ++i) {
		cumVal += area[i] / areaSum * count;
		int cumRounded = ei::round(cumVal);
		// The min(count-1) is necessary to avoid a child cell which itself
		// already has the split count -> would lead to a dead lock.
		//int subCount = ei::min(count - 1, cumRounded - prevCumRounded); // More correct
		int subCount = ei::clamp(cumRounded - prevCumRounded, minCount, static_cast<int>(count - 1));
		//int subCount = minCount;
		// TODO
		m_nodes[children + i] = static_cast<DataType>(subCount);
		prevCumRounded = cumRounded;
	}
}

template class DmOctree<std::int32_t>;
template class DmOctree<float>;

} // namespace mufflon::data_structs