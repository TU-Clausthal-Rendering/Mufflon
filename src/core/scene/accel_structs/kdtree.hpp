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

	void split(int* splitDim, int* refDim, int num, int center, int dimToSplit, std::vector<int>& tmp) {
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

	void query_euclidean_rec(const Vec& refPosition, int k, int* idx, float* distSq, int c) const {
		if(c == -1) return;
		const Vec pos = m_positions[m_tree[c].data];
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
		float hyperPlaneDist = refPosition[m_tree[c].splitDim] - pos[m_tree[c].splitDim];
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