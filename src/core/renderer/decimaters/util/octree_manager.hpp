#pragma once

#include "octree.hpp"
#include "util/types.hpp"
#include "core/memory/unique_device_ptr.hpp"
#include <ei/3dtypes.hpp>
#include <memory>
#include <vector>

namespace mufflon { namespace renderer { namespace decimaters {

template < class T >
class OctreeManager {
public:
	using OctreeType = T;
	using ReadOnlyType = ReadOnlyOctree<OctreeType>;
	using NodeType = typename OctreeType::NodeType;

	OctreeManager(const u32 capacity, const u32 octreeCapacity) :
		m_capacity{ 1u + ((static_cast<std::size_t>(capacity) + 7u) & ~7) },
		m_allocationCounter{ 0u },
		m_nodeMemory{ std::make_unique<std::atomic<NodeType>[]>(m_capacity) },
		m_octrees{},
		m_cudaNodeMemory{ make_udevptr_array<Device::CUDA, NodeType, false>(m_capacity) },
		m_cudaOctrees{ nullptr }
	{
		m_octrees.reserve(octreeCapacity);
	}
	OctreeManager(const OctreeManager&) = delete;
	OctreeManager(OctreeManager&& other) noexcept :
		m_capacity{ other.m_capacity },
		m_allocationCounter{ other.m_allocationCounter.load() },
		m_nodeMemory{ std::move(other.m_nodeMemory) },
		m_octrees{ std::move(other.m_octrees) },
		m_cudaNodeMemory{ std::move(other.m_cudaNodeMemory) },
		m_cudaOctrees{ std::move(other.m_cudaOctrees) }
	{}
	OctreeManager& operator=(const OctreeManager&) = delete;
	OctreeManager& operator=(OctreeManager&&) = delete;
	~OctreeManager() = default;

	template < class... Args >
	void create(const ei::Box& bounds, Args&& ...args) {
		if(m_allocationCounter.load() > 9u * m_octrees.size())
			throw std::runtime_error("Creating new octrees is only allowed before any inserts happen!");
		m_octrees.emplace_back(bounds, static_cast<u32>(m_capacity), m_nodeMemory.get(),
							   m_nodeMemory[m_allocationCounter.fetch_add(1u)], m_allocationCounter,
							   std::forward<Args>(args)...);
	}
	void update_readonly() {
		std::vector<ReadOnlyType> readOnlyTrees;
		readOnlyTrees.reserve(m_octrees.size());
		m_cudaOctrees = make_udevptr_array<Device::CUDA, ReadOnlyType, false>(m_octrees.size());
		static_assert(sizeof(*m_nodeMemory.get()) == sizeof(*m_cudaNodeMemory.get()),
					  "Mismatch in node memory sizes (atomic not equal to non-atomic)");
		copy(reinterpret_cast<unsigned char*>(m_cudaNodeMemory.get()),
			 reinterpret_cast<unsigned char*>(m_nodeMemory.get()),
			 m_allocationCounter.load() * sizeof(*m_cudaNodeMemory.get()));
		for(const auto& octree : m_octrees)
			readOnlyTrees.emplace_back(octree.inverted_diagonal(), octree.minimum_bound(),
									   m_cudaNodeMemory, m_cudaNodeMemory[octree.root_offset()],
									   octree.depth());
		copy(m_cudaOctrees.get(), readOnlyTrees.data(), sizeof(readOnlyTrees.front()) * readOnlyTrees.size());
	}

	const ReadOnlyType* get_cuda_readonly_trees() const noexcept { return m_cudaOctrees.get(); }
	OctreeType* data() noexcept { return m_octrees.data(); }
	const OctreeType* data() const noexcept { return m_octrees.data(); }
	OctreeType& operator[](const std::size_t index) noexcept { return m_octrees[index]; }
	const OctreeType& operator[](const std::size_t index) const noexcept { return m_octrees[index]; }

	typename std::vector<OctreeType>::iterator begin() noexcept { return m_octrees.begin(); }
	typename std::vector<OctreeType>::iterator end() noexcept { return m_octrees.end(); }
	typename std::vector<OctreeType>::const_iterator begin() const noexcept { return m_octrees.begin(); }
	typename std::vector<OctreeType>::const_iterator end() const noexcept { return m_octrees.end(); }
	typename std::vector<OctreeType>::const_iterator cbegin() const noexcept { return m_octrees.cbegin(); }
	typename std::vector<OctreeType>::const_iterator cend() const noexcept { return m_octrees.cend(); }

	std::size_t capacity() const noexcept { return m_capacity; }
	std::size_t size() const noexcept { return m_allocationCounter.load(); }
	bool empty() const noexcept { return size() == 0u; }

private:
	std::size_t m_capacity;
	std::atomic_size_t m_allocationCounter;
	std::unique_ptr<std::atomic<NodeType>[]> m_nodeMemory;
	std::vector<OctreeType> m_octrees;

	// CUDA-things
	unique_device_ptr<Device::CUDA, NodeType[]> m_cudaNodeMemory;
	unique_device_ptr<Device::CUDA, ReadOnlyType[]> m_cudaOctrees;
};

}}} // namespace mufflon::renderer::decimaters