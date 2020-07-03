#pragma once

#include "octree_nodes.hpp"
#include "util/assert.hpp"
#include "util/int_types.hpp"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <array>
#include <atomic>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace mufflon { namespace renderer { namespace decimaters {

template < class N >
class Octree {
public:
	using NodeType = N;

	struct NodeIndex {
		u32 index;
		u32 depthMask;
	};

	Octree(const ei::Box& bounds, u32 capacity, u32 fillCapacity, std::atomic<NodeType>* nodes,
		   std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter);
	Octree(const Octree&) = delete;
	Octree(Octree&& other) noexcept;
	Octree& operator=(const Octree&) = delete;
	Octree& operator=(Octree&&) = delete;
	~Octree() = default;

	NodeIndex get_node_index(const ei::Vec3& pos) const noexcept;
	std::optional<NodeIndex> get_node_index(const ei::Vec3& pos, const std::vector<bool>& stopMask) const noexcept;
	float get_samples(const ei::Vec3& pos) const noexcept;
	float get_samples(const NodeIndex index) const noexcept;
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal) const noexcept;
	void add_sample(const ei::Vec3& pos, const ei::Vec3& normal, const float value) noexcept;

	std::pair<std::vector<std::pair<u32, float>>, std::size_t> to_grid(const std::size_t maxDepth) const noexcept;
	void export_to_file(const std::string& path, const std::size_t maxDepth = std::numeric_limits<std::size_t>::max()) const;
	double compute_leaf_sum() const noexcept;

	const std::atomic<NodeType>& root() const noexcept { return m_root; }
	NodeIndex root_index() const noexcept {
		return NodeIndex{
			static_cast<u32>(&m_root - m_nodes),
			1u << m_depth.load(std::memory_order_acquire)
		};
	}
	const std::atomic<NodeType>* nodes() const noexcept { return m_nodes; }
	const std::atomic<NodeType>& node(const std::size_t index) const noexcept {
		mAssert(index < m_allocationCounter.load(std::memory_order_acquire));
		return m_nodes[index];
	}
	std::optional<std::array<NodeIndex, 8u>> children(const NodeIndex index) const noexcept;
	float get_inverse_cell_volume(const NodeIndex index) const noexcept {
		const auto gridRes = 1u << m_depth.load(std::memory_order_acquire);
		const auto currRes = gridRes / index.depthMask;
		return static_cast<float>(currRes) * ei::prod(m_diagonalInv);
	}

	std::size_t capacity() const noexcept { return m_capacity; }
	std::size_t fill_capacity() const noexcept { return m_fillCapacity; }
	std::size_t leafs() const noexcept { return m_childCounter.load(); }
	const ei::Vec3& diagonal() const noexcept { return m_diagonal; }
	const ei::Vec3& inverted_diagonal() const noexcept { return m_diagonalInv; }
	const ei::Vec3& minimum_bound() const noexcept { return m_minBound; }
	std::ptrdiff_t root_offset() const noexcept { return m_nodes - &m_root; }
	u32 depth() const noexcept { return m_depth.load(); }

protected:
	// Increments the node value or splits it if necessary, and
	// returns if the increment should be performed again in a later
	// iteration (simply reuse curr pointer, which will then point
	// to a new parent node)
	virtual std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
													  const ei::UVec3& gridPos, const ei::Vec3& offPos,
													  const float value, const u32 depth) noexcept = 0;

	// Initializes the children after a split based on a distribution of surface area
	virtual void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
										const ei::Vec3& offPos, const std::size_t depth,
										const NodeType& parentValue, const u32 childOffset) = 0;

	ei::Vec3 m_diagonal;
	ei::Vec3 m_diagonalInv;
	ei::Vec3 m_minBound;
	std::size_t m_capacity;
	std::size_t m_fillCapacity;
	std::atomic<NodeType>* m_nodes;
	std::atomic<NodeType>& m_root;
	std::atomic_size_t& m_allocationCounter;
	std::atomic_size_t m_childCounter;
	std::atomic_uint32_t m_depth;
	std::atomic_bool m_stopSplitting;
};

class FloatOctree final : public Octree<FloatOctreeNode> {
public:
	using NodeType = FloatOctreeNode;

	FloatOctree(const ei::Box& bounds, u32 capacity, u32 fillCapacity, std::atomic<NodeType>* nodes,
						 std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter,
						 const float splitVal);
	FloatOctree(const FloatOctree&) = delete;
	FloatOctree(FloatOctree&& other) noexcept;
	FloatOctree& operator=(const FloatOctree&) = delete;
	FloatOctree& operator=(FloatOctree&&) = delete;
	virtual ~FloatOctree() = default;
	
	template < class O >
	void join(const Octree<O>& other, const float weight) noexcept;

protected:
	std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
												  const ei::UVec3& gridPos, const ei::Vec3& offPos,
												  const float value, const u32 depth) noexcept final;

	void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
								const ei::Vec3& offPos, const std::size_t depth,
								const NodeType& parentValue, const u32 childOffset) final;

private:
	float m_splitViewVal;
};


class SampleOctree final : public Octree<SampleOctreeNode> {
public:
	using NodeType = SampleOctreeNode;

	SampleOctree(const ei::Box& bounds, u32 capacity, u32 fillCapacity, std::atomic<NodeType>* nodes,
				 std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter,
				 const u32 splitCount, const float splitVal);
	SampleOctree(const SampleOctree&) = delete;
	SampleOctree(SampleOctree&& other) noexcept;
	SampleOctree& operator=(const SampleOctree&) = delete;
	SampleOctree& operator=(SampleOctree&&) = delete;
	virtual ~SampleOctree() = default;

protected:
	std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
												  const ei::UVec3& gridPos, const ei::Vec3& offPos,
												  const float value, const u32 depth) noexcept final;

	void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
								const ei::Vec3& offPos, const std::size_t depth,
								const NodeType& parentValue, const u32 childOffset) final;

private:
	u32 m_splitCount;
	float m_splitValue;
};

}}} // namespace mufflon::renderer::decimaters