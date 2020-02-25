#include "octree.inl"

namespace mufflon::renderer::decimaters {


__host__ FloatOctree::FloatOctree(const ei::Box& bounds, u32 capacity, std::atomic<NodeType>* nodes,
								  std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter,
								  const float splitVal) :
	Octree<FloatOctreeNode>{ bounds, capacity, nodes, root, allocationCounter },
	m_splitViewVal{ splitVal }
{}

__host__ FloatOctree::FloatOctree(FloatOctree&& other) noexcept :
	Octree<FloatOctreeNode>{ std::move(other) },
	m_splitViewVal{ other.m_splitViewVal }
{}

__host__ std::atomic<FloatOctreeNode>* FloatOctree::add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
																  const ei::UVec3& gridPos, const ei::Vec3& offPos,
																  const float value, const u32 depth) noexcept {
	// Make sure we don't overflow the depth
	if(depth >= std::numeric_limits<decltype(depth)>::digits - 2) {
		m_stopSplitting = true;
		return nullptr;
	}

	// Try to add a sample to the node
	auto oldNodeVal = curr->load(std::memory_order_acquire);
	NodeType newNodeVal;
	do {
		// Check if the node is (or has turned into) a parent - in this case abort
		if(oldNodeVal.is_parent())
			return curr;
		newNodeVal = NodeType::as_split_child(oldNodeVal.get_value() + value);
	} while(!curr->compare_exchange_weak(oldNodeVal, newNodeVal, std::memory_order_acq_rel, std::memory_order_acquire));

	// If we did increment but stopped filling/splitting, then we simply do nothing
	if(m_stopSplitting)
		return nullptr;

	// Cannot check for equality like with single integer increments - we only know that we're over the line
	if(newNodeVal.get_value() >= m_splitViewVal &&
	   newNodeVal.get_value() < std::numeric_limits<float>::max()) {
		// CAS to determine which thread performs the split
		// Has to be strong to avoid spurious failure (won't happen on x86, but still)
		if(curr->compare_exchange_strong(newNodeVal, NodeType::as_split_child(std::numeric_limits<float>::max()),
										 std::memory_order_acq_rel, std::memory_order_acquire)) {
			// We are now the splitting thread - allocate new children
			const auto offset = m_allocationCounter.fetch_add(8u, std::memory_order::memory_order_consume);
			// Ensure that we don't overflow our node array
			if(offset + 8u >= m_capacity) {
				// TODO: to avoid a deadlock here we reset the node count to zero, even though it should
				// theoretically remain the samples; this may introduce some bias!
				m_allocationCounter.store(m_capacity);
				m_stopSplitting = true;
				curr->store(NodeType::as_split_child(0.f), std::memory_order_release);
				return nullptr;
			}
			m_childCounter += 7u;

			// Initialize children
			init_children(normal, gridPos, offPos, depth, newNodeVal, static_cast<u32>(offset));
			curr->store(NodeType::as_parent(static_cast<u32>(offset)), std::memory_order_release);

			auto oldDepth = m_depth.load(std::memory_order::memory_order_acquire);
			decltype(oldDepth) newDepth;
			do {
				newDepth = ei::max(oldDepth, depth + 1u);
			} while(!m_depth.compare_exchange_weak(oldDepth, newDepth, std::memory_order_release));
			// If we split, then we do not perform the insertion again
			return nullptr;
		} else {
			// ´Spin until split occurred
			while(!curr->load(std::memory_order_acquire).is_parent_or_fresh()) {}
		}
	}
	// If we only incremented or spun after increment, then we wanna redo the increment in greater
	// depth, if available
	return curr;
}

__host__ void FloatOctree::init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
										 const ei::Vec3& offPos, const std::size_t depth,
										 const NodeType& parentValue, const u32 childOffset) {
	const ei::Vec3 childCellSize = m_diagonal / (1u << (depth + 1u));
	const ei::Vec3 localPos = offPos - 2.f * childCellSize * gridPos;
	// Get the intersection areas of the eight children to distribute
	// the count properly.
	float area[8];
	float areaSum = 0.0f;
	for(int i = 0; i < 8; ++i) {
		const ei::IVec3 childLocalPos{ i & 1, (i >> 1) & 1, i >> 2 };
		// TODO
		area[i] = 1.f;// math::intersection_area_nrm(childCellSize, localPos - childLocalPos * childCellSize, normal);
		areaSum += area[i];
	}

	for(u32 i = 0u; i < 8u; ++i)
		m_nodes[childOffset + i].store(NodeType::as_split_child(area[i] * parentValue.get_value() / areaSum), std::memory_order_release);
}

} // namespace mufflon::renderer::decimaters