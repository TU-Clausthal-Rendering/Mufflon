#pragma once

#include "octree.hpp"
#include "util/log.hpp"

namespace mufflon { namespace renderer { namespace decimaters {

template < class N >
__host__ Octree<N>::Octree(const ei::Box& bounds, u32 capacity, std::atomic<N>* nodes,
				std::atomic<N>& root, std::atomic_size_t& allocationCounter) :
	m_diagonal{ (bounds.max - bounds.min) * 1.002f },
	m_diagonalInv{ 1.f / m_diagonal },
	m_minBound{ bounds.min - m_diagonal * (1.002f - 1.f) / 2.f },
	m_capacity{ capacity },
	m_nodes{ nodes },
	m_root{ root },
	m_allocationCounter{ allocationCounter },
	m_childCounter{ 8u },
	m_depth{ 1u },
	m_stopSplitting{ false }
{
	// Initialize the root node and its children
	const auto offset = m_allocationCounter.fetch_add(8u, std::memory_order_seq_cst);
	m_root.store(N::as_parent(static_cast<u32>(offset)), std::memory_order_release);
	if(!std::atomic<FloatOctreeNode>{}.is_lock_free())
		logWarning("[ViewOctree::ViewOctree] Atomic operations for 8-byte are not lock-free on your platform");
	for(std::size_t i = 0u; i < 8u; ++i)
		m_nodes[offset + i] = N{};
}

template < class N >
__host__ Octree<N>::Octree(Octree<N>&& other) noexcept :
	m_diagonal{ other.m_diagonal },
	m_diagonalInv{ other.m_diagonalInv },
	m_minBound{ other.m_minBound },
	m_capacity{ other.m_capacity },
	m_nodes{ other.m_nodes },
	m_root{ other.m_root },
	m_allocationCounter{ other.m_allocationCounter },
	m_childCounter{ other.m_childCounter.load() },
	m_depth{ other.m_depth.load() },
	m_stopSplitting{ other.m_stopSplitting.load() }
{}

template < class N >
__host__ float Octree<N>::get_samples(const ei::Vec3& pos) const noexcept {
	const ei::Vec3 offPos = pos - m_minBound;
	const ei::Vec3 normPos = offPos * m_diagonalInv;
	// Get the integer position on the finest level.
	const decltype(m_depth.load()) gridRes = 1u << m_depth.load(std::memory_order_acquire);
	const ei::UVec3 iPos{ normPos * gridRes };
	auto currentLvlMask = gridRes;
	auto currVal = m_root.load(std::memory_order_consume);
	while(currVal.is_parent()) {
		currentLvlMask >>= 1;
		const auto offset = currVal.get_child_offset();
		// Get the relative index of the child [0,7] plus the child offset for the node index
		const auto idx = ((iPos.x & currentLvlMask) ? 1 : 0)
			+ ((iPos.y & currentLvlMask) ? 2 : 0)
			+ ((iPos.z & currentLvlMask) ? 4 : 0)
			+ offset;
		currVal = m_nodes[idx].load(std::memory_order_acquire);
	}
	return currVal.get_sample();
}

template < class N >
__host__ void Octree<N>::add_sample(const ei::Vec3& pos, const ei::Vec3& normal, const float value) noexcept {
	const ei::Vec3 offPos = pos - m_minBound;
	const ei::Vec3 normPos = offPos * m_diagonalInv;
	const ei::UVec3 iPos{ normPos * (1u << 30u) };
	decltype(m_depth.load()) lvl = 1u;
	auto* curr = &m_root;
	auto currVal = curr->load(std::memory_order_consume);
	do {
		const auto currOffset = currVal.get_child_offset();
		ei::UVec3 gridPos = iPos >> (30u - lvl);
		const int idx = (gridPos.x & 1) + 2 * (gridPos.y & 1) + 4 * (gridPos.z & 1)
			+ currOffset;
		auto* next = &m_nodes[idx];
		// Increment and split if suitable (if it's a leaf)
		curr = add_and_split(next, normal, gridPos, offPos, value, lvl);
		lvl += 1u;
		if(curr == nullptr)
			break;
		currVal = curr->load(std::memory_order_acquire);
	} while(currVal.is_parent());
}

template < class N >
__host__ std::pair<std::vector<std::pair<u32, float>>, std::size_t> Octree<N>::to_grid(const std::size_t maxDepth) const noexcept {
	std::vector<std::pair<const std::atomic<N>*, ei::UVec4>> queue;
	queue.reserve(this->leafs());
	queue.emplace_back(&m_root, ei::UVec4{ 0u, 0u, 0u, 0u });

	// Allocate the grid
	const std::size_t depth = std::min(maxDepth, static_cast<std::size_t>(m_depth.load()));
	const std::size_t res = 1llu << depth;
	std::vector<std::pair<u32, float>> grid(res * res * res, std::make_pair(0u, 0.f));

	while(!queue.empty()) {
		const auto curr = queue.back();
		queue.pop_back();
		const auto currVal = curr.first->load(std::memory_order_acquire);
		if(currVal.is_parent()) {
			const auto offset = currVal.get_child_offset();
			const ei::UVec3 base{
				2u * curr.second.x,
				2u * curr.second.y,
				2u * curr.second.z,
			};
			for(u32 i = 0u; i < 8u; ++i) {
				queue.emplace_back(&m_nodes[offset + i], ei::UVec4{
					base.x + (i & 1),
					base.y + ((i & 2) >> 1u),
					base.z + (i >> 2u),
					curr.second.w + 1u
				});
			}
		} else {
			// We need to fill entire ranges at the given depth
			std::size_t missedLevels = std::max(maxDepth, static_cast<std::size_t>(curr.second.w)) - maxDepth;
			const auto size = 1llu << (depth - std::min(depth, static_cast<std::size_t>(curr.second.w)));
			ei::UVec3 coord{ curr.second };
			if(missedLevels != 0u)
				coord >>= missedLevels;

			const auto startX = static_cast<std::size_t>(coord.x) * size;
			const auto startY = static_cast<std::size_t>(coord.y) * size;
			const auto startZ = static_cast<std::size_t>(coord.z) * size;
			const auto endX = startX + size;
			const auto endY = startY + size;
			const auto endZ = startZ + size;

			for(std::size_t x = startX; x < endX; ++x) {
				for(std::size_t y = startY; y < endY; ++y) {
					for(std::size_t z = startZ; z < endZ; ++z) {
						const auto index = x + res * y + z * res * res;
						grid.at(index) = std::make_pair(curr.second.w, currVal.get_sample());
					}
				}
			}
		}
	}

	return std::make_pair(grid, res);
}

template < class N >
__host__ void Octree<N>::export_to_file(const std::string& path, const std::size_t maxDepth) const {
	const auto [grid, res] = this->to_grid(maxDepth);
	gli::texture3d texture{ gli::format::FORMAT_RGBA8_UNORM_PACK8,
							gli::ivec3{ res, res, res }, 1u };

	// Find max. first
	float maxOctreeVal = 0.f;
	for(std::size_t i = 0u; i < res * res * res; ++i) {
		if(maxOctreeVal < grid[i].second)
			maxOctreeVal = grid[i].second;
	}

	for(std::size_t z = 0u; z < res; ++z) {
		for(std::size_t y = 0u; y < res; ++y) {
			for(std::size_t x = 0u; x < res; ++x) {
				const auto gridIndex = x + y * res + z * res * res;
				const auto color = static_cast<u8>(255.f * grid[gridIndex].second / maxOctreeVal);
				if(color != 0.f)
					texture.store(gli::ivec3{ x, y, z }, 0u, gli::u8vec4{ color, static_cast<u8>(grid[gridIndex].first), 0.f, 255.f });
				else
					texture.store(gli::ivec3{ x, y, z }, 0u, gli::u8vec4{ 0.f, 0.f, 0.f, 0.f });
			}
		}
	}
	gli::save(texture, path);
}

}}} // namespace mufflon::renderer::decimaters