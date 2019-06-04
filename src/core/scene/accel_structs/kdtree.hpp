#pragma once

#include <ei/vector.hpp>
#include <memory>
#include <atomic>
#include <algorithm>
#include <vector>

namespace mufflon::scene::accel_struct {

/* General kd-tree container on CPU Device (TODO: GPU version? acquire()?).
 * Uses median splits to partitionate a set of points which was added iteratively before.
 * Usage:
 *		1. Insert data multithreaded (lockfree). The data must fit into a predefined capacity.
 *		2. Call build()
 *		3. Use range queries.
 *
 * TODO: provide a 'parent' array for BVH builders?
 */
template < typename Data, int N >
class KdTree {
public:
	using Vec = ei::Vec<float, N>;
	// 'Remove' all data from the map O(1)
	void clear() {
		m_dataCount.store(0);
	}

	// Recreate memory if different from current capacity. Clears the container.
	void reserve(int numExpectedEntries) {
		if(m_dataCapacity != numExpectedEntries) {
			m_dataCapacity = numExpectedEntries;
			m_data = std::make_unique<Data[]>(numExpectedEntries);
			m_positions = std::make_unique<Vec[]>(numExpectedEntries);
			m_tree = std::make_unique<Node[]>(numExpectedEntries);
		}
		clear();
	}

	int size() const { return m_dataCount.load(); }
	int capacity() const { return m_dataCapacity; }
	int mem_size() const { return m_dataCapacity * (sizeof(Data) + sizeof(Vec) + sizeof(Node)); }

	Data* insert(const Vec& position, const Data& data) { return insert(position, Data{data}); }
	Data* insert(const Vec& position, Data&& data) {
		int dataIdx = m_dataCount.fetch_add(1);
		// TODO: more robust overflow behavior?
		m_data[dataIdx] = std::move(data);
		m_positions[dataIdx] = position;
		return &m_data[dataIdx];
	}

	Data& get_data_by_index(int index) { return m_data[index]; }
	const Data& get_data_by_index(int index) const { return m_data[index]; }

	// This build method is based on an inverted merge sort.
	// It starts with a sorted array in each dimension and then recursively
	// splits the tree at the median element of the largest dimension.
	void build() {
		int n = m_dataCount.load();
		// Create a sorted array along each data axis
		std::unique_ptr<int[]> sorted[N];
		for(int dim = 0; dim < N; ++dim) {
			sorted[dim] = std::make_unique<int[]>(n);
			for(int i = 0; i < n; ++i)
				sorted[dim][i] = i;
			std::sort( &sorted[dim][0], &sorted[dim][n],
				[this,dim](const int l, const int r) { return m_positions[l][dim] < m_positions[r][dim]; }
			);
		}

		// Call recursive build kernel (kind of inverse merge sort)
		BuildInfo info;
		info.n = n;
		info.currentCounter = 0;
		info.sorted = sorted;
		info.tmp.resize(n / 2 + 1);
		build(info, 0, n);

		resort_positions();
	}

	// This build method uses a kind of quicksort with median of N elements.
	// Instead of sorting the elements in each subtree, it only moves them on the
	// left/right side.
	void build2() {
		// Create two working index buffers (ping pong) and get the bounding box.
		Vec bbMin{ m_positions[0] };
		Vec bbMax{ m_positions[0] };
		int n = m_dataCount.load();
		std::unique_ptr<int[]> tmp[2];
		tmp[0] = std::make_unique<int[]>(n);
		for(int i = 0; i < n; ++i) {
			tmp[0][i] = i;
			bbMin = min(bbMin, m_positions[i]);
			bbMax = max(bbMax, m_positions[i]);
		}
		tmp[1] = std::make_unique<int[]>(n); // No need for initilization

		int allocCounter = 0;
		build_qs(tmp[0].get(), tmp[1].get(), bbMin, bbMax, n, allocCounter);

		resort_positions();
	}

	// Find the nearest k points within a hypersphere around refPosition.
	// idx: memory to take k indices. These can be used with get_data_by_index
	//		to access the neighbors
	// distSq: memory to take the squared distances to the k points.
	void query_euclidean(const Vec& refPosition, int k, int* idx, float* distSq) const {
		mAssert(k < m_dataCount.load());
		for(int i = 0; i < k; ++i)
			distSq[i] = std::numeric_limits<float>::infinity();
		query_euclidean_rec(refPosition, k, idx, distSq, 0);
	}

	/*bool tree_is_sane() const {
		m_tree[0].data *= -1;
		for(int i = 0; i < m_dataCount; ++i) {
			m_tree[m_tree[i].left].data *= -1;
			m_tree[m_tree[i].right].data *= -1;
		}
	}*/

	int compute_depth(int node = 0) const {
		if(node == -1) return 0;
		return 1 + ei::max(compute_depth(m_tree[node].left), compute_depth(m_tree[node].right));
	}

private:
	struct Node {
		int left {-1};
		int right {-1};
		int data {-1};
		int splitDim {0};
	};
	std::unique_ptr<Data[]> m_data;
	std::unique_ptr<Vec[]> m_positions;
	std::unique_ptr<Node[]> m_tree;
	int m_dataCapacity { 0 };
	std::atomic_int32_t m_dataCount { 0 };

	struct BuildInfo {
		int n;									// Number of data entries = number of nodes
		int currentCounter;						// Index of the next free node
		const std::unique_ptr<int[]>* sorted;	// One array per dimension
		std::vector<int> tmp;					// Temporary memory for the split operation
	};
	// c: target index of the current node.
	// l: first index in the range of nodes (inclusive).
	// r: index at the end of the range (exclusive).
	void build(BuildInfo& info, int l, int r) {
		// Find dimension with the largest extent
		Vec min, max;
		for(int dim = 0; dim < N; ++dim) {
			min[dim] = m_positions[info.sorted[dim][l]][dim];
			max[dim] = m_positions[info.sorted[dim][r-1]][dim];
		}
		Vec extent = max - min;
		int splitDim = 0;
		for(int dim = 1; dim < N; ++dim)
			if(extent[dim] > extent[splitDim]) splitDim = dim;

		// Split at median
		int m = (l + r) / 2;
		// The split requires to reorder all other dimension arrays
		for(int dOff = 1; dOff < N; ++dOff) {
			// Mark all points right of the median with the same value by increasing them arbitrary.
			// The value is reset to the median after the split.
			/*const Vec& refPos = m_positions[info.sorted[splitDim][m]];
			int j = m+1;
			while(m_positions[info.sorted[splitDim][j]][splitDim] == refPos[splitDim]) {
				m_positions[info.sorted[splitDim][j]][splitDim] = 1e38f;
				++j;
			}*/
			int coDim = (splitDim + dOff) % N;
			split(&info.sorted[coDim][l], &info.sorted[splitDim][l],
				r - l, m - l, splitDim, info.tmp);
			// Reset the offset.
			/*--j;
			while(j > m) {
				m_positions[info.sorted[splitDim][j]][splitDim] = refPos[splitDim];
				--j;
			}*/
		}

		int c = info.currentCounter++;
		m_tree[c].data = info.sorted[splitDim][m];
		m_tree[c].splitDim = splitDim;
		// Recursive build children. The confusing counter stuff is to increase the data locality.
		// TODO: check if locality is worth something, or if we can simply use the counter in the
		// above line directly (no c at all).
		// It is possible that a right child is empty because we store data on all levels.
		// If there is only one element it got stored in the current node
		if(m > l) {
			m_tree[c].left = info.currentCounter;
			build(info, l, m);
		} else m_tree[c].left = -1;
		// If there are two elements the second is stored in this node and the first gets the left child.
		if(m+1 < r) {
			m_tree[c].right = info.currentCounter;
			build(info, m+1, r);
		} else m_tree[c].right = -1;
	}

	void split(int* __restrict splitDim, int* __restrict refDim, int num, int center, int dimToSplit, std::vector<int>& tmp) {
		Vec refPos = m_positions[refDim[center]];
		// Setup two pointers to the two new range begins
		int l = 0;
		int r = 1; // Relative to center
		for(int i = 0; i < num; ++i) {
			// Find out in which of the two sets the current point belongs.
			// 1. Special case: the center point itself.
			if(splitDim[i] == refDim[center]) tmp[0] = splitDim[i];
			else {
				// 2. Use the position to find out where the datum lies.
				Vec pos = m_positions[splitDim[i]];
				if(pos[dimToSplit] < refPos[dimToSplit]) splitDim[l++] = splitDim[i];
				else if(pos[dimToSplit] > refPos[dimToSplit]) tmp[r++] = splitDim[i];
				else {
					// Not sure: the coordinate is the same as that of the center,
					// so we have to search the index in the refDim.
					for(int j = center + 1; j < num; ++j) { // Towards right
						if(refDim[j] == splitDim[i]) {
							tmp[r++] = splitDim[i];
							goto FOUND;
						}
					}
					for(int j = center - 1; j >= 0; --j) { // Towards left
						if(refDim[j] == splitDim[i]) {
							splitDim[l++] = splitDim[i];
							goto FOUND;
						}
					}
					mAssertMsg(false, "Element is not part of both dimensions!?");
				FOUND:;
				}
			}
		}
		// Copy complete solution back
		memcpy( splitDim + center, &tmp[0], r * sizeof(int) );
	}


	// Improve cache performance for queries by resorting the position array
	void resort_positions() {
		auto newPositions = std::make_unique<Vec[]>(m_dataCapacity);
		int n = m_dataCount.load();
		for(int i = 0; i < n; ++i) {
			newPositions[i] = m_positions[m_tree[i].data];
		}
		std::swap(m_positions, newPositions);
	}


	// Find the median via selection sort.
	int median(int* candidates, int count, int splitDim) {
		if(count <= 2) return candidates[0];
		int m = count / 2;
/*		for(int i = 0; i <= m; ++i) for(int j = i+1; j < count; ++j)
			if(m_positions[canditates[j]][splitDim] < m_positions[canditates[i]][splitDim])
				std::swap(canditates[i], canditates[j]);*/
		std::nth_element(candidates, candidates + m, candidates + count,
			[this,splitDim](const int a, const int b){ return m_positions[a][splitDim] < m_positions[b][splitDim]; } );
		return candidates[m];
	}

	int closest_to(const int* candidates, int count, int splitDim, float wantedPos) {
		float minD = ei::abs(m_positions[candidates[0]][splitDim] - wantedPos);
		int minC = 0;
		for(int i = 1; i < count; ++i) {
			float d = ei::abs(m_positions[candidates[i]][splitDim] - wantedPos);
			if(d < minD) {
				minD = d;
				minC = i;
			}
		}
		return candidates[minC];
	}

	int build_qs(int* __restrict indices, int* __restrict tmp, const Vec& bbMin, const Vec& bbMax, int count, int& allocCounter) {
		// "allocate" a new node
		int node = allocCounter++;

		// No need for splitting
		if(count == 1) {
			m_tree[node].data = indices[0];
			m_tree[node].splitDim = 0;
			m_tree[node].left = -1;
			m_tree[node].right = -1;
			return node;
		}

		// Determine the largest dimension for splitting
		Vec bbTmp = bbMax - bbMin;
		int splitDim = 0;
		for(int dim = 1; dim < N; ++dim)
			if(bbTmp[dim] > bbTmp[splitDim]) splitDim = dim;

		// Get median of some random elements
		constexpr int MEDIAN_C = 9;
		int candidates[MEDIAN_C];
		int step = ei::max(1, count / MEDIAN_C);
		int medianCount = ei::min(count, MEDIAN_C);
		int off = (count - (medianCount - 1) * step) / 2;
		for(int i = 0; i < medianCount; ++i)
			candidates[i] = indices[off + i * step];
		// Find the median via selection sort up to the element M.
		//int refIdx = median(candidates, medianCount, splitDim);
		int refIdx = closest_to(candidates, medianCount, splitDim, bbMin[splitDim] + bbTmp[splitDim] * 0.5f);
		float refPos = m_positions[refIdx][splitDim];

		// Split the data into the two sets
		int pl = 0;
		int pr = count-1;
		bool putEqualLeft = true;
		for(int i = 0; i < count; ++i) {
			if(indices[i] != refIdx) { // The reference element must not be in any of the two sets
				float curPos = m_positions[indices[i]][splitDim];
				if(curPos < refPos || (putEqualLeft && curPos == refPos))
					tmp[pl++] = indices[i];
				else
					tmp[pr--] = indices[i];
				if(curPos == refPos) putEqualLeft = !putEqualLeft;
			}
		}
		++pr; // Let cr point to the first element on the right side

		m_tree[node].data = refIdx;
		m_tree[node].splitDim = splitDim;
		// Recursive build. Note that tmp and indices are swapped, because tmp now contains
		// our valid node sets and the previous indices can be used as temporary memory.
		if(pl > 0) {
			bbTmp = bbMax; bbTmp[splitDim] = refPos;
			m_tree[node].left = build_qs(tmp, indices, bbMin, bbTmp, pl, allocCounter);
		} else m_tree[node].left = -1;
		if(count-pr > 0) {
			bbTmp = bbMin; bbTmp[splitDim] = refPos;
			m_tree[node].right = build_qs(tmp+pr, indices+pr, bbTmp, bbMax, count-pr, allocCounter);
		} else m_tree[node].right = -1;

		return node;
	}

	void query_euclidean_rec(const Vec& refPosition, int k, int* idx, float* distSq, int c) const {
		if(c == -1) return;
		//const Vec& pos = m_positions[m_tree[c].data];
		const Vec& pos = m_positions[c];
		float hyperPlaneDist = refPosition[m_tree[c].splitDim] - pos[m_tree[c].splitDim];
		float cDistSq = lensq(refPosition - pos);
		// Insertion sort in the canditate array.
		if(cDistSq < distSq[k-1]) {
			int i = k - 1;	// i will be the insertion position
			while(i > 0 && cDistSq < distSq[i-1]) {
				idx[i] = idx[i-1];
				distSq[i] = distSq[i-1];
				--i;
			}
			idx[i] = m_tree[c].data;
			distSq[i] = cDistSq;
		}

		// Search candidates in the subtrees.
		if(hyperPlaneDist <= 0.0f) {
			// Search first on the left side
			query_euclidean_rec(refPosition, k, idx, distSq, m_tree[c].left);
			// Prune tree if possible
			if(hyperPlaneDist * hyperPlaneDist < distSq[k-1])
				query_euclidean_rec(refPosition, k, idx, distSq, m_tree[c].right);
		} else {
			// Search first on the right
			query_euclidean_rec(refPosition, k, idx, distSq, m_tree[c].right);
			// Prune tree if possible
			if(hyperPlaneDist * hyperPlaneDist < distSq[k-1])
				query_euclidean_rec(refPosition, k, idx, distSq, m_tree[c].left);
		}
	}
};

} // namespace mufflon::scene::accel_struct