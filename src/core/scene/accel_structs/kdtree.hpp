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

	int insert(const Vec& position, const Data& data) { return insert(position, Data{data}); }
	int insert(const Vec& position, Data&& data) {
		int dataIdx = m_dataCount.fetch_add(1);
		// TODO: more robust overflow behavior?
		m_data[dataIdx] = std::move(data);
		m_positions[dataIdx] = position;
		return dataIdx;
	}

	Data& get_data_by_index(int index) { return m_data[index]; }
	const Data& get_data_by_index(int index) const { return m_data[index]; }
	Vec& get_position_by_index(int index) { return m_positions[index]; }
	const Vec& get_position_by_index(int index) const { return m_positions[index]; }

	// This build method uses a kind of quicksort with median of N elements.
	// Instead of sorting the elements in each subtree, it only moves them on the
	// left/right side.
	void build() {
		// Create two working index buffers (ping pong) and get the bounding box.
		Vec bbMin{ m_positions[0] };
		Vec bbMax{ m_positions[0] };
		int n = m_dataCount.load();
		for(int i = 0; i < n; ++i) {
			bbMin = min(bbMin, m_positions[i]);
			bbMax = max(bbMax, m_positions[i]);
		}

		int allocCounter = 0;
		build_qs(bbMin, bbMax, 0, n, allocCounter);
	}

	// Find the nearest k points within a hypersphere around refPosition.
	// idx: memory to take k indices. These can be used with get_data_by_index
	//		to access the neighbors
	// distSq: memory to take the squared distances to the k points.
	void query_euclidean(const Vec& refPosition, int k, int* idx, float* distSq,
						 float maxDistSq = std::numeric_limits<float>::infinity()) const {
		mAssert(k < m_dataCount.load());
		for(int i = 0; i < k; ++i) {
			distSq[i] = maxDistSq;
			idx[i] = -1;
		}
		query_euclidean_rec(refPosition, k, idx, distSq, 0);
	}


	int compute_depth(int node = 0) const {
		if(node == -1) return 0;
		return 1 + ei::max(compute_depth(m_tree[node].left), compute_depth(m_tree[node].right));
	}

private:
	struct Node {
		int left {-1};
		int right {-1};
		int splitDim {0};
	};
	std::unique_ptr<Data[]> m_data;
	std::unique_ptr<Vec[]> m_positions;
	std::unique_ptr<Node[]> m_tree;
	int m_dataCapacity { 0 };
	std::atomic_int32_t m_dataCount { 0 };


	// Find the median via selection sort.
	std::pair<float,int> median(std::pair<float,int>* candidates, int count, int splitDim) {
		if(count <= 2) return candidates[0];
		int m = count / 2;
/*		for(int i = 0; i <= m; ++i) for(int j = i+1; j < count; ++j)
			if(m_positions[canditates[j]][splitDim] < m_positions[canditates[i]][splitDim])
				std::swap(canditates[i], canditates[j]);*/
		std::nth_element(candidates, candidates + m, candidates + count,
			[this,splitDim](const std::pair<float,int> a, const std::pair<float,int> b){ return m_positions[a.second][splitDim] < m_positions[b.second][splitDim]; } );
		return candidates[m];
	}

	std::pair<float,int> closest_to(const std::pair<float,int>* candidates, int count, int splitDim, float wantedPos) {
		float minD = ei::abs(candidates[0].first - wantedPos);
		int minC = 0;
		for(int i = 1; i < count; ++i) {
			float d = ei::abs(candidates[i].first - wantedPos);
			if(d < minD) {
				minD = d;
				minC = i;
			}
		}
		return candidates[minC];
	}

	int build_qs(const Vec& bbMin, const Vec& bbMax, int left, int count, int& allocCounter) {
		// "allocate" a new node
		int node = allocCounter++;

		// No need for splitting
		if(count == 1) {
			m_tree[node].splitDim = 0;
			m_tree[node].left = -1;
			m_tree[node].right = -1;
			return node;
		}
		if(count == 2) {
			m_tree[node].splitDim = 0;
			m_tree[node].left = -1;
			m_tree[node].right = -1;
			int otherNode = allocCounter++;
			m_tree[otherNode].splitDim = 0;
			m_tree[otherNode].left = -1;
			m_tree[otherNode].right = -1;
			if(m_positions[left][0] <= m_positions[left+1][0])
				m_tree[node].right = otherNode;
			else
				m_tree[node].left = otherNode;
			return node;
		}

		// Determine the largest dimension for splitting
		Vec bbTmp = bbMax - bbMin;
		int splitDim = 0;
		for(int dim = 1; dim < N; ++dim)
			if(bbTmp[dim] > bbTmp[splitDim]) splitDim = dim;

		// Get median of some random elements
		constexpr int MEDIAN_C = 9;
		std::pair<float,int> candidates[MEDIAN_C];	// TODO: stack saving optimization possible
		int step = ei::max(1, count / MEDIAN_C);
		int medianCount = ei::min(count, MEDIAN_C);
		int off = left + (count - (medianCount - 1) * step) / 2;
		for(int i = 0; i < medianCount; ++i) {
			int iiIdx = off + i * step;
			mAssert(iiIdx < left + count);
			candidates[i] = {m_positions[iiIdx][splitDim], iiIdx};
		}
		// Find the median via selection sort up to the element M.
		auto ref = median(candidates, medianCount, splitDim);
		//auto ref = closest_to(candidates, medianCount, splitDim, bbMin[splitDim] + bbTmp[splitDim] * 0.5f);

		// Swap the refIdx element to the tree-node index (improves caching which is
		// important for the position array.
		// Further, it is not necessary to store a data index in the tree nodes, improving
		// the cache performance further and reducing memory consumption.
		std::swap(m_positions[ref.second], m_positions[node]);
		std::swap(m_data[ref.second], m_data[node]);
		mAssert(node == left);

		// Split the data into the two sets (quicksort partitioning)
		int pl = left + 1; // The reference element must not be in any of the two sets. Above swap + i=1 skip the reference element.
		int pr = left + count-1;
		bool putEqualLeft = true;
		while(pl < pr) {
			while(pl < pr && (m_positions[pl][splitDim] <= ref.first)) {
				if(m_positions[pl][splitDim] == ref.first) {
					if(putEqualLeft)
						putEqualLeft = !putEqualLeft;
					else break;
				}
				++pl;
			}
			while(pl < pr && (m_positions[pr][splitDim] >= ref.first)) {
				if(m_positions[pr][splitDim] == ref.first) {
					if(!putEqualLeft)
						putEqualLeft = !putEqualLeft;
					else break;
				}
				--pr;
			}
			if(pl < pr) {
				std::swap(m_positions[pl], m_positions[pr]);
				std::swap(m_data[pl], m_data[pr]);
				++pl;
				--pr;
			}
		}
		// Either pl or pl+1 is the split position.
		if(m_positions[pl][splitDim] < ref.first)
			++pl;
		mAssert(((pl == left + count) || m_positions[pl][splitDim] >= ref.first) && m_positions[pl-1][splitDim] <= ref.first);
		mAssert(pl <= left + count);

		m_tree[node].splitDim = splitDim;
		// Recursive build. Note that tmp and indices are swapped, because tmp now contains
		// our valid node sets and the previous indices can be used as temporary memory.
		int cl = pl - left - 1;
		int cr = count - cl - 1;
		if(cl > 0) {
			bbTmp = bbMax; bbTmp[splitDim] = ref.first;
			m_tree[node].left = build_qs(bbMin, bbTmp, left+1, cl, allocCounter);
		} else m_tree[node].left = -1;
		if(cr > 0) {
			bbTmp = bbMin; bbTmp[splitDim] = ref.first;
			m_tree[node].right = build_qs(bbTmp, bbMax, pl, cr, allocCounter);
		} else m_tree[node].right = -1;

		return node;
	}

	void query_euclidean_rec(const Vec& refPosition, int k, int* idx, float* distSq, int c) const {
		if(c == -1) return;
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
			idx[i] = c;
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