#pragma once

#include "octree.hpp"

namespace mufflon { namespace renderer { namespace decimaters {

template < class O >
__host__ void FloatOctree::join(const Octree<O>& other, const float weight) noexcept {
	// TODO: what should happen in case of fill stop?

	// Perform DFS(-ish)
	// Those nodes being children in both trees are simply added with weight into our tree
	std::vector<std::tuple<std::atomic<NodeType>*, const std::atomic<O>*, std::size_t>> queue;
	std::vector<std::pair<std::atomic<NodeType>*, std::size_t>> subQueue;
	std::vector<std::tuple<std::atomic<NodeType>*, const std::atomic<O>*, std::size_t>> subTreeQueue;
	// TODO: more intelligent reserving?
	queue.reserve(ei::max(leafs(), other.leafs()));
	subQueue.reserve(leafs());
	subTreeQueue.reserve(leafs());
	queue.emplace_back(&m_root, &other.root(), 0u);

	while(!queue.empty()) {
		const auto current = queue.back();
		queue.pop_back();

		const auto currOur = std::get<0>(current)->load(std::memory_order_acquire);
		const auto currOther = std::get<1>(current)->load(std::memory_order_acquire);

		// Check status
		if(currOur.is_parent() && currOther.is_parent()) {
			// Nothing to do except add the children to the queue
			const auto ourOffset = currOur.get_child_offset();
			const auto otherOffset = currOther.get_child_offset();
			for(int i = 0; i < 8; ++i)
				queue.emplace_back(&m_nodes[ourOffset + i], &other.node(otherOffset + i), std::get<2>(current) + 1u);
		} else if(currOur.is_leaf() && currOther.is_leaf()) {
			// TODO: split!
			// Add the weighted value to our leaf and terminate this traversal path
			const auto newVal = currOur.get_sample() + weight * currOther.get_sample();
			if(newVal >= m_splitViewVal) {
				const auto offset = m_allocationCounter.fetch_add(8u);
				mAssert(offset < m_capacity);
				m_childCounter += 7u;
				const auto newNode = NodeType::as_parent(static_cast<u32>(offset));
				std::get<0>(current)->store(newNode, std::memory_order_release);
				for(u32 i = 0u; i < 8u; ++i)
					m_nodes[offset + i] = NodeType::as_split_child(newVal / 8.f);
				m_depth.store(ei::max(m_depth.load(std::memory_order_acquire),
									   static_cast<u32>(std::get<2>(current)) + 1u),
							  std::memory_order_release);
			} else {
				const auto newValue = NodeType::as_split_child(newVal);
				std::get<0>(current)->store(newValue, std::memory_order_release);
			}
		} else if(currOur.is_leaf() && currOther.is_parent()) {
			// We have to insert the sub-tree into our own and volume-weightedly add the
			// node value to the sub-tree children
			const auto baseValue = currOur.get_sample();
			subTreeQueue.clear();
			subTreeQueue.emplace_back(std::get<0>(current), std::get<1>(current), 0u);

			while(!subTreeQueue.empty()) {
				const auto curr = subTreeQueue.back();
				subTreeQueue.pop_back();
				const auto otherVal = std::get<1>(curr)->load(std::memory_order_acquire);

				if(otherVal.is_parent()) {
					// Allocate the necessary nodes
					const auto otherOffset = otherVal.get_child_offset();
					const auto offset = m_allocationCounter.fetch_add(8u);
					m_childCounter += 7u;
					const auto newNode = NodeType::as_parent(static_cast<u32>(offset));
					std::get<0>(curr)->store(newNode, std::memory_order_release);
					m_depth.store(ei::max(m_depth.load(std::memory_order_acquire),
										  static_cast<u32>(std::get<2>(curr)) + 1u),
								  std::memory_order_release);
					// We don't have to initialize the children here, since later iterations will do that
					for(u32 i = 0u; i < 8u; ++i)
						subTreeQueue.emplace_back(&m_nodes[offset + i], &other.node(otherOffset + i), std::get<2>(curr) + 1u);
				} else {
					// Found a leaf, store the appropriate value in the accompanied (own) leaf
					const auto divisor = static_cast<float>(1u << (3u * std::get<2>(curr)));
					const auto newVal = baseValue / divisor + weight * otherVal.get_sample();
					if(newVal >= m_splitViewVal) {
						const auto offset = m_allocationCounter.fetch_add(8u);
						m_childCounter += 7u;
						const auto newNode = NodeType::as_parent(static_cast<u32>(offset));
						std::get<0>(curr)->store(newNode, std::memory_order_release);
						for(u32 i = 0u; i < 8u; ++i)
							m_nodes[offset + i] = NodeType::as_split_child(newVal / 8.f);
						m_depth.store(ei::max(m_depth.load(std::memory_order_acquire),
											   static_cast<u32>(std::get<2>(curr)) + 1u),
									  std::memory_order_release);
					} else {
						const auto newNode = NodeType::as_split_child(newVal);
						std::get<0>(curr)->store(newNode, std::memory_order_release);
					}
				}
			}
		} else {
			// Only thing left: we are parent and they are leaf -> distribute value across children
			// This happens in a density-conserving matter: we know that the volume of the
			// children has to sum up to the volume of the parent
			const auto weightedValue = weight * currOther.get_sample();
			subQueue.clear();
			subQueue.emplace_back(std::get<0>(current), 0u);

			while(!subQueue.empty()) {
				const auto curr = subQueue.back();
				subQueue.pop_back();
				const auto subVal = curr.first->load(std::memory_order_acquire);

				if(subVal.is_parent()) {
					// Queue the children
					const auto offset = subVal.get_child_offset();
					for(u32 i = 0u; i < 8u; ++i)
						subQueue.push_back(std::make_pair(&m_nodes[offset + i], curr.second + 1u));
				} else {
					// Add split amount
					const auto share = weightedValue / static_cast<float>(1u << (3u * curr.second));
					const auto newValue = NodeType::as_split_child(subVal.get_sample() + share);
					curr.first->store(newValue, std::memory_order_release);
				}
			}
		}
	}
}

}}} // namespace mufflon::renderer::decimaters