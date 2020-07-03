#pragma once

#include "util/assert.hpp"
#include "util/types.hpp"
#include "core/math/intersection_areas.hpp"
#include <ei/3dtypes.hpp>
#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

namespace mufflon { namespace data_structs {


class CountOctree {
public:
	struct Data {
		int count;
		float value;
	};

	struct NodeId {
		u32 levelMask;
		u32 index;
	};

	CountOctree(const CountOctree&) = delete;
	CountOctree(CountOctree&& other) noexcept :
		m_diagonal{ other.m_diagonal },
		m_diagonalInv{ other.m_diagonalInv },
		m_minBound{ other.m_minBound },
		m_splitCount{ other.m_splitCount },
		m_capacity{ other.m_capacity },
		m_nodes{ std::move(other.m_nodes) },
		m_root{ other.m_root },
		m_allocationCounter{ other.m_allocationCounter },
		m_childCounter{ other.m_childCounter.load() },
		m_depth{ other.m_depth.load() },
		m_stopSplitting{ other.m_stopSplitting }
	{}
	CountOctree& operator=(const CountOctree&) = delete;
	CountOctree& operator=(CountOctree&&) = default;
	~CountOctree() = default;

	Data sum() const noexcept {
		std::vector<const Node*> queue;
		queue.reserve(m_capacity);
		float sum = 0.f;
		u32 count = 0u;
		queue.push_back(&m_root);
		while(!queue.empty()) {
			const Node* curr = queue.back();
			queue.pop_back();
			if(curr->is_parent()) {
				const auto offset = curr->get_child_offset();
				for(u32 i = 0u; i < 8u; ++i)
					queue.push_back(&m_nodes[offset + i]);
			} else {
				const auto data = curr->get_data();
				if(data.count > 0u) {
					sum += data.value;
					count += static_cast<u32>(data.count);
				}
			}
		}
		return Data{ static_cast<int>(count), sum };
	}

	// Utility functions for external traversal
	const NodeId get_root_node() const noexcept {
		return NodeId{ 1u << m_depth.load(), static_cast<u32>(&m_root - m_nodes) };
	}
	bool is_leaf(const NodeId& id) const noexcept {
		mAssert(id.index < m_allocationCounter.load());
		return m_nodes[id.index].is_leaf();
	}
	std::array<NodeId, 8u> get_children(const NodeId& id) const noexcept {
		mAssert(id.index < m_allocationCounter.load());
		const auto& node = m_nodes[id.index];
		mAssert(node.is_parent());
		const auto offset = node.get_child_offset();
		return { {
			NodeId{ id.levelMask >> 1u, offset + 0u },
			NodeId{ id.levelMask >> 1u, offset + 1u },
			NodeId{ id.levelMask >> 1u, offset + 2u },
			NodeId{ id.levelMask >> 1u, offset + 3u },
			NodeId{ id.levelMask >> 1u, offset + 4u },
			NodeId{ id.levelMask >> 1u, offset + 5u },
			NodeId{ id.levelMask >> 1u, offset + 6u },
			NodeId{ id.levelMask >> 1u, offset + 7u }
		} };
	}

	NodeId get_node_id(const ei::Vec3& pos) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		const auto gridRes = 1u << m_depth.load();
		const ei::UVec3 iPos{ normPos * gridRes };
		// Get root value. This will most certainly be a child pointer...
		const Node* node = &m_root;
		// The most significant bit in iPos distinguishes the children of the root node.
		// For each level, the next bit will be the relevant one.
		auto currentLvlMask = gridRes;
		u32 idx = 0u;
		while(node->is_parent()) {
			currentLvlMask >>= 1;
			const auto offset = node->get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			node = &m_nodes[idx];
		}
		return NodeId{ currentLvlMask, idx };
	}

	NodeId get_node_id(const ei::Vec3& pos, const std::vector<bool>& nodeMask) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		const auto gridRes = 1u << m_depth.load();
		const ei::UVec3 iPos{ normPos * gridRes };
		// Get root value. This will most certainly be a child pointer...
		const Node* node = &m_root;
		// The most significant bit in iPos distinguishes the children of the root node.
		// For each level, the next bit will be the relevant one.
		auto currentLvlMask = gridRes;
		u32 idx = 0u;
		while(node->is_parent() && !nodeMask[node - m_nodes]) {
			currentLvlMask >>= 1;
			const auto offset = node->get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			node = &m_nodes[idx];
		}
		return NodeId{ currentLvlMask, idx };
	}

	void add_sample(const ei::Vec3& pos, const ei::Vec3& normal, const float value) noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		const ei::UVec3 iPos{ normPos * (1u << 30u) };
		decltype(m_depth.load()) lvl = 1u;
		Node* curr = &m_root;
		do {
			const auto currOffset = curr->get_child_offset();
			ei::UVec3 gridPos = iPos >> (30u - lvl);
			const int idx = (gridPos.x & 1) + 2 * (gridPos.y & 1) + 4 * (gridPos.z & 1)
				+ currOffset;
			Node* next = &m_nodes[idx];
			// Increment and split if suitable (if it's a leaf)
			curr = add_and_split(*next, normal, gridPos, offPos, value, lvl);
			lvl += 1u;
		} while((curr != nullptr) && curr->is_parent());
	}

	Data get_samples(const ei::Vec3& pos) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		const decltype(m_depth.load()) gridRes = 1u << m_depth.load();
		const ei::UVec3 iPos{ normPos * gridRes };
		// Get root value. This will most certainly be a child pointer...
		const Node* node = &m_root;
		// The most significant bit in iPos distinguishes the children of the root node.
		// For each level, the next bit will be the relevant one.
		auto currentLvlMask = gridRes;
		while(node->is_parent()) {
			currentLvlMask >>= 1;
			const auto offset = node->get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			const auto idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			node = &m_nodes[idx];
		}

		return node->get_data();
	}
	Data get_samples(const NodeId& id) const noexcept {
		mAssert(id.index < m_capacity);
		mAssert(id.levelMask <= (1u << m_depth.load()));
		return m_nodes[id.index].get_data();
	}
	
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal, const bool ignoreCount = false) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		const decltype(m_depth.load()) gridRes = 1u << m_depth.load();
		const ei::UVec3 iPos{ normPos * gridRes };
		// Get root value. This will most certainly be a child pointer...
		const Node* node = &m_root;
		// The most significant bit in iPos distinguishes the children of the root node.
		// For each level, the next bit will be the relevant one.
		auto currentLvlMask = gridRes;
		while(node->is_parent()) {
			currentLvlMask >>= 1;
			const auto offset = node->get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			const auto idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			node = &m_nodes[idx];
		}

		const auto data = node->get_data();
		if(data.count > 0u) {
			currentLvlMask = ei::max(1u, currentLvlMask);
			const auto currentGridRes = gridRes / currentLvlMask;
			const ei::UVec3 cellPos = iPos / currentLvlMask;
			const ei::Vec3 cellSize = 1.0f / (currentGridRes * m_diagonalInv);
			const ei::Vec3 cellMin = cellPos * cellSize;
			const float area = math::intersection_area_nrm(cellSize, offPos - cellMin, normal);
			if(ignoreCount)
				return sdiv(data.value, area);
			else
				return sdiv(data.value, static_cast<float>(data.count) * area);
		}
		return 0.f;// data.value;
	}
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal, const NodeId& id,
					  const bool ignoreCount = false) const noexcept {
		mAssert(id.index < m_capacity);
		mAssert(id.levelMask <= (1u << m_depth.load()));
		const auto data = m_nodes[id.index].get_data();
		if(data.count > 0u) {
			const ei::Vec3 offPos = pos - m_minBound;
			const ei::Vec3 normPos = offPos * m_diagonalInv;
			// Get the integer position on the finest level.
			const auto gridRes = 1u << m_depth.load();
			const ei::UVec3 iPos{ normPos * gridRes };

			const auto currentLvlMask = ei::max(1u, id.levelMask);
			const auto currentGridRes = gridRes / currentLvlMask;
			const ei::UVec3 cellPos = iPos / currentLvlMask;
			const ei::Vec3 cellSize = 1.0f / (currentGridRes * m_diagonalInv);
			const ei::Vec3 cellMin = cellPos * cellSize;
			const float area = math::intersection_area_nrm(cellSize, offPos - cellMin, normal);
			if(ignoreCount)
				return sdiv(data.value, area);
			else
				return sdiv(data.value, static_cast<float>(data.count) * area);
		}
		return 0.f;
	}

#if 0
	float get_interpolated_density(const ei::Vec3& pos, const ei::Vec3& normal, const bool ignoreCount = false) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		const auto maxLvl = m_depth.load();
		const auto gridRes = 1 << (maxLvl + 1);
		const ei::IVec3 iPos{ normPos * gridRes };
		// Memory to track nodes
		u32 buffer[16];
		float areaBuffer[16];
		u32* parents = buffer;
		u32* current = buffer + 8;
		float* parentArea = areaBuffer;
		float* currentArea = areaBuffer + 8;
		for(u32 i = 0u; i < 8u; ++i) {
			current[i] = 0;	// Initialize to root level
			currentArea[i] = 0.0f;
		}
		decltype(m_depth.load()) lvl = 0;
		ei::IVec3 parentMinPos{ 0u };
		const Node* node = &m_root;
		bool anyHadChildren = node->is_parent();
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
			const ei::IVec3 nextLvlPos = iPos >> (maxLvl - lvl);	// Next level coordinate
			const ei::IVec3 lvlPos = nextLvlPos / 2 - 1 + (nextLvlPos & 1);	// Min coordinate of the 8 cells on next level
			int lvlRes = 1 << lvl;
			const ei::Vec3 cellSize = m_diagonal / lvlRes;
			anyHadChildren = false;	// Check for the new children inside the for loop
			for(int i = 0; i < 8; ++i) {
				const ei::IVec3 cellPos = lvlPos + CELL_ITER[i];
				// We need to find the parent in the 'parents' buffer array.
				// Since the window of interpolation moves the reference coordinate
				// we subtract 'parentMinPos' scaled to the current level.
				const ei::IVec3 localParent = (cellPos - parentMinPos) / 2;
				mAssert(localParent >= 0 && localParent <= 1);
				const u32 parentIdx = localParent.x + 2 * localParent.y + 4 * localParent.z;
				// Check if parent node has children.
				const u32 parentAddress = parents[parentIdx];
				const Node* curr = &m_nodes[parentAddress];
				if(curr->is_parent()) {
					// Insert the child node's address
					const u32 localChildIdx = (cellPos.x & 1) + 2 * (cellPos.y & 1) + 4 * (cellPos.z & 1);
					current[i] = curr->get_child_offset() + localChildIdx;
					//currentArea[i] = -1.0f;
					const auto isParent = m_nodes[current[i]].is_parent();
					anyHadChildren |= isParent;
					// Compute the area if this is a leaf node
					if(!isParent) {
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
		const auto lvlRes = 1u << lvl;
		const ei::Vec3 tPos = normPos * lvlRes - 0.5f;
		const ei::IVec3 gridPos = ei::floor(tPos);
		ei::Vec3 ws[2];
		ws[1] = tPos - gridPos;
		ws[0] = 1.0f - ws[1];
		float countSum = 0.0f, areaSum = 0.0f, valueSum = 0.f;
		const ei::Vec3 cellSize{ m_diagonal / lvlRes };
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
			mAssert(m_nodes[current[i]].is_leaf());
			if(area > 0.0f) {
				const float w = ws[CELL_ITER[i].x].x * ws[CELL_ITER[i].y].y * ws[CELL_ITER[i].z].z;
				const auto data = m_nodes[current[i]].get_data();
				countSum += data.count * w * lvlFactor;
				areaSum += area * w;
				valueSum += data.value * w * lvlFactor;
			}
		}
		mAssert(areaSum > 0.0f);
		if(ignoreCount)
			return sdiv(valueSum, areaSum);
		else
			return sdiv(valueSum, areaSum * countSum);
	}
#endif // 0

	std::size_t capacity() const noexcept {
		return m_capacity;
	}
	std::size_t children() const noexcept {
		return m_childCounter.load();
	}

	const ei::Vec3& get_minimum_boundary() const noexcept { return m_minBound; }

	const ei::Vec3 get_bounds_diagonal() const noexcept {
		return m_diagonal;
	}
	const ei::Vec3 get_bounds_inverted_diagonal() const noexcept {
		return m_diagonalInv;
	}

	u32 get_root_depth_mask() const noexcept {
		return 1u << m_depth.load();
	}

private:
	friend class CountOctreeManager;

	static constexpr ei::IVec3 CELL_ITER[8] = {
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
	};

	class Node {
	public:
		struct SampleAddResult {
			bool performed;
			u32 countOrOffset;
			float value;
		};

		Node() = default;
		Node(const Node& other) noexcept : m_data{ other.m_data.load() } {}
		Node(Node&& other) noexcept : m_data{ other.m_data.load() } {}
		Node& operator=(const Node& other) noexcept {
			m_data.store(other.m_data.load());
			return *this;
		}
		Node& operator=(Node&& other) noexcept {
			m_data.store(other.m_data.load());
			return *this;
		}
		~Node() = default;

		// Creates a node that is pointing to other nodes as a parent
		static Node as_parent(u32 offset) noexcept {
			return Node{ -static_cast<int>(offset), 0.f };
		}
		static Node as_split_child(u32 initCount, float initValue) noexcept {
			return Node{ static_cast<int>(initCount), initValue };
		}

		bool is_parent() const noexcept { return m_data.load(std::memory_order_acquire).count < 0; }
		bool is_leaf() const noexcept { return !is_parent(); }

		// This function is purely for the spinlock functionality in case
		// the capacity limit has been reached
		bool is_parent_or_fresh() const noexcept { return m_data.load(std::memory_order_acquire).count <= 0; }

		u32 get_child_offset() const noexcept {
			mAssert(this->is_parent());
			return static_cast<u32>(-m_data.load(std::memory_order_acquire).count);
		}

		Data get_data() const noexcept {
			mAssert(this->is_leaf());
			return m_data.load(std::memory_order_acq_rel);
		}

		SampleAddResult add_sample(float value) noexcept {
			auto oldV = m_data.load(std::memory_order_acquire);
			Data newV;
			do {
				if(oldV.count < 0) return { false, static_cast<u32>(-oldV.count), 0.f };
				newV = Data{ oldV.count + 1, oldV.value + value };
			} while(!m_data.compare_exchange_weak(oldV, newV, std::memory_order::memory_order_acq_rel));
			return { true, static_cast<u32>(newV.count), newV.value };
		}

	private:
		Node(int countOrOffset, float value) : m_data{ Data{ countOrOffset, value } } {}

		std::atomic<Data> m_data{};
	};

	CountOctree(const ei::Box& bounds, u32 capacity, u32 splitCount, Node* nodes, Node& root,
				std::atomic_size_t& allocationCounter) :
		m_diagonal{ (bounds.max - bounds.min) * 1.002f },
		m_diagonalInv{ 1.f / m_diagonal },
		m_minBound{ bounds.min - m_diagonal * (1.002f - 1.f) / 2.f },
		m_splitCount{ splitCount },
		m_capacity{ capacity },
		m_nodes{ nodes },
		m_root{ root },
		m_allocationCounter{ allocationCounter },
		m_childCounter{ 8u },
		m_depth{ 1u },
		m_stopSplitting{ false }
	{
		// Initialize the root node and its children
		const auto offset = m_allocationCounter.fetch_add(8u);
		m_root = Node::as_parent(static_cast<u32>(offset));
		if(!std::atomic<Data>{}.is_lock_free())
			logWarning("[CountOctree::CountOctree] Atomic operations for 8-byte are not lock-free on your platform");
		for(std::size_t i = 0u; i < 8u; ++i)
			m_nodes[offset + i] = Node{};
	}

	// Increments the node value or splits it if necessary, and
	// returns if the increment should be performed again in a later
	// iteration (simply reuse curr pointer, which will then point
	// to a new parent node)
	Node* add_and_split(Node& curr, const ei::Vec3& normal, const ei::UVec3& gridPos,
						const ei::Vec3& offPos, const float value, const u32 depth) noexcept {
		// Make sure we don't overflow the depth
		if(depth >= std::numeric_limits<decltype(depth)>::digits - 2) {
			m_stopSplitting = true;
			return nullptr;
		}

		// Try to add a sample to the node
		const auto result = curr.add_sample(value);
		// If we didn't increment it we go to the next node (must have been a parent)
		if(!result.performed)
			return &curr;

		// If we did increment but stopped filling/splitting, then we simply do nothing
		if(m_stopSplitting)
			return nullptr;

		if(result.countOrOffset == m_splitCount) {
			// Allocate new children
			// Allocate new children
			const auto offset = m_allocationCounter.fetch_add(8u, std::memory_order::memory_order_acq_rel);
			// Ensure that we don't overflow our node array
			if(offset + 8 >= m_capacity) {
				// TODO: to avoid a deadlock here we reset the node count to zero, even though it should
				// theoretically remain the samples; this may introduce some bias!
				m_allocationCounter.store(m_capacity);
				m_stopSplitting = true;
				curr = Node{};
				return nullptr;
			}
			m_childCounter += 7u;
			// Initialize children
			init_children(normal, gridPos, offPos, depth, result.countOrOffset, result.value, static_cast<u32>(offset));
			for(std::size_t i = 0u; i < 8u; ++i)
				m_nodes[offset + i] = Node::as_split_child(static_cast<u32>(m_splitCount / 8.f), result.value / 8.f);
			curr = Node::as_parent(static_cast<u32>(offset));

			auto oldDepth = m_depth.load(std::memory_order::memory_order_acquire);
			decltype(oldDepth) newDepth;
			do {
				newDepth = std::max(oldDepth, depth + 1u);
			} while(!m_depth.compare_exchange_weak(oldDepth, newDepth, std::memory_order::memory_order_seq_cst));
			// If we split, then we do not perform the insertion again
			return nullptr;
		} else if(result.countOrOffset > m_splitCount) {
			// Spin until cell is split
			while(!curr.is_parent_or_fresh()) {}
		}
		// If we only incremented or spun after increment, then we wanna redo the increment in greater
		// depth, if available
		return &curr;
	}

	// Initializes the children after a split based on a distribution of surface area
	void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
					   const ei::Vec3& offPos, const std::size_t depth,
					   const u32 parentCount, const float parentValue, const u32 childOffset) {
		const ei::Vec3 childCellSize = m_diagonal / (1u << (depth + 1u));
		const ei::Vec3 localPos = offPos - 2.f * childCellSize * gridPos;
		// Get the intersection areas of the eight children to distribute
		// the count properly.
		float area[8];
		float areaSum = 0.0f;
		for(int i = 0; i < 8; ++i) {
			const ei::IVec3 childLocalPos{ i & 1, (i >> 1) & 1, i >> 2 };
			area[i] = math::intersection_area_nrm(childCellSize, localPos - childLocalPos * childCellSize, normal);
			areaSum += area[i];
		}
		
		const u32 minCount = ei::ceil(parentCount / 8.0f);
		// Distribute the count proportional to the areas. To avoid loss we cannot
		// simply round. https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
		float cumVal = 0.f;
		u32 prevCumRounded = 0;
		for(int i = 0; i < 8; ++i) {
			cumVal += area[i] / areaSum * parentCount;
			const u32 cumRounded = ei::round(cumVal);
			// The min(count-1) is necessary to avoid a child cell which itself
			// already has the split count -> would lead to a dead lock.
			//int subCount = ei::min(count - 1, cumRounded - prevCumRounded); // More correct
			const u32 subCount = ei::clamp(cumRounded - prevCumRounded, minCount, static_cast<u32>(parentCount - 1));
			const auto share = static_cast<float>(subCount) * parentValue / static_cast<float>(parentCount);
			m_nodes[childOffset + i] = Node::as_split_child(subCount, share);
			prevCumRounded = cumRounded;
		}
	}

	ei::Vec3 m_diagonal;
	ei::Vec3 m_diagonalInv;
	ei::Vec3 m_minBound;
	u32 m_splitCount;
	std::size_t m_capacity;
	Node* m_nodes;
	Node& m_root;
	std::atomic_size_t& m_allocationCounter;
	std::atomic_size_t m_childCounter;
	std::atomic_uint32_t m_depth;
	bool m_stopSplitting;
};

class CountOctreeManager {
public:
	CountOctreeManager(const u32 capacity, const u32 octreeCapacity, const u32 splitCount) :
		m_capacity{ 1u + ((static_cast<std::size_t>(capacity) + 7u) & ~7) },
		m_allocationCounter{ 0u },
		m_nodeMemory{ std::make_unique<CountOctree::Node[]>(m_capacity) },
		m_octrees{},
		m_splitCount{ splitCount }
	{
		m_octrees.reserve(octreeCapacity);
	}

	void create(const ei::Box& bounds) {
		if(m_allocationCounter.load() > 9u * m_octrees.size())
			throw std::runtime_error("Creating new octrees is only allowed before any inserts happen!");
		m_octrees.push_back(CountOctree{ bounds, static_cast<u32>(m_capacity), m_splitCount, m_nodeMemory.get(),
							m_nodeMemory[m_allocationCounter.fetch_add(1u)], m_allocationCounter });
	}

	CountOctree& operator[](const std::size_t index) noexcept { return m_octrees[index]; }
	const CountOctree& operator[](const std::size_t index) const noexcept { return m_octrees[index]; }

	std::vector<CountOctree>::iterator begin() noexcept { return m_octrees.begin(); }
	std::vector<CountOctree>::iterator end() noexcept { return m_octrees.end(); }
	std::vector<CountOctree>::const_iterator begin() const noexcept { return m_octrees.begin(); }
	std::vector<CountOctree>::const_iterator end() const noexcept { return m_octrees.end(); }
	std::vector<CountOctree>::const_iterator cbegin() const noexcept { return m_octrees.cbegin(); }
	std::vector<CountOctree>::const_iterator cend() const noexcept { return m_octrees.cend(); }

	std::size_t capacity() const noexcept { return m_capacity; }
	std::size_t size() const noexcept { return m_allocationCounter.load(); }
	bool empty() const noexcept { return size() == 0u; }

private:
	std::size_t m_capacity;
	std::atomic_size_t m_allocationCounter;
	std::unique_ptr<CountOctree::Node[]> m_nodeMemory;
	std::vector<CountOctree> m_octrees;
	u32 m_splitCount;
};

}} // namespace mufflon::data_structs