#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "core/export/core_api.h"
#include <ei/3dtypes.hpp>
#include <ei/vector.hpp>
#include <cuda_runtime.h>
#include <atomic>
#include <limits>
#include <string>
#include <vector>

namespace mufflon { namespace renderer { namespace decimaters {

template < class N >
class Octree {
public:
	using NodeType = N;

	__host__ Octree(const ei::Box& bounds, u32 capacity, std::atomic<NodeType>* nodes,
					std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter);
	Octree(const Octree&) = delete;
	__host__ Octree(Octree&& other) noexcept;
	Octree& operator=(const Octree&) = delete;
	Octree& operator=(Octree&&) = delete;
	~Octree() = default;

	__host__ float get_samples(const ei::Vec3& pos) const noexcept;
	__host__ void add_sample(const ei::Vec3& pos, const ei::Vec3& normal, const float value) noexcept;
	__host__ std::pair<std::vector<std::pair<u32, float>>, std::size_t> to_grid(const std::size_t maxDepth) const noexcept;
	__host__ void export_to_file(const std::string& path, const std::size_t maxDepth = std::numeric_limits<std::size_t>::max()) const;

	const std::atomic<NodeType>& root() const noexcept { return m_root; }
	const std::atomic<NodeType>* nodes() const noexcept { return m_nodes; }
	const std::atomic<NodeType>& node(const std::size_t index) const noexcept { return m_nodes[index]; }

	__host__ std::size_t capacity() const noexcept { return m_capacity; }
	__host__ std::size_t leafs() const noexcept { return m_childCounter.load(); }
	__host__ const ei::Vec3& diagonal() const noexcept { return m_diagonal; }
	__host__ const ei::Vec3& inverted_diagonal() const noexcept { return m_diagonalInv; }
	__host__ const ei::Vec3& minimum_bound() const noexcept { return m_minBound; }
	__host__ std::ptrdiff_t root_offset() const noexcept { return m_nodes - &m_root; }
	__host__ u32 depth() const noexcept { return m_depth.load(); }

protected:
	// Increments the node value or splits it if necessary, and
	// returns if the increment should be performed again in a later
	// iteration (simply reuse curr pointer, which will then point
	// to a new parent node)
	virtual __host__ std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
													  const ei::UVec3& gridPos, const ei::Vec3& offPos,
													  const float value, const u32 depth) noexcept = 0;

	// Initializes the children after a split based on a distribution of surface area
	virtual __host__ void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
										const ei::Vec3& offPos, const std::size_t depth,
										const NodeType& parentValue, const u32 childOffset) = 0;

	ei::Vec3 m_diagonal;
	ei::Vec3 m_diagonalInv;
	ei::Vec3 m_minBound;
	std::size_t m_capacity;
	std::atomic<NodeType>* m_nodes;
	std::atomic<NodeType>& m_root;
	std::atomic_size_t& m_allocationCounter;
	std::atomic_size_t m_childCounter;
	std::atomic_uint32_t m_depth;
	std::atomic_bool m_stopSplitting;
};

template < class O >
class ReadOnlyOctree {
public:
	using OctreeType = O;
	using NodeType = typename OctreeType::NodeType;

	__host__ ReadOnlyOctree(const ei::Vec3& invDiag, const ei::Vec3& minBound,
								 const NodeType* nodes, const NodeType& root,
								 const u32 depth) noexcept :
		m_diagonalInv{ invDiag },
		m_minBound{ minBound },
		m_nodes{ nodes },
		m_root{ root },
		m_depth{ depth }
	{}
	ReadOnlyOctree(const ReadOnlyOctree&) = default;
	ReadOnlyOctree(ReadOnlyOctree&&) = default;
	ReadOnlyOctree& operator=(const ReadOnlyOctree&) = delete;
	ReadOnlyOctree& operator=(ReadOnlyOctree&&) = delete;
	~ReadOnlyOctree() = default;

	__host__ __device__ float get_samples(const ei::Vec3& pos) const noexcept {
		const ei::Vec3 offPos = pos - m_minBound;
		const ei::Vec3 normPos = offPos * m_diagonalInv;
		// Get the integer position on the finest level.
		// Get/set of samples never happens at the same time, so having no barriers is fine
		const decltype(m_depth) gridRes = 1u << m_depth;
		const ei::UVec3 iPos{ normPos * gridRes };
		auto currentLvlMask = gridRes;
		auto currVal = m_root;
		while(currVal.is_parent()) {
			currentLvlMask >>= 1;
			const auto offset = currVal.get_child_offset();
			// Get the relative index of the child [0,7] plus the child offset for the node index
			const auto idx = ((iPos.x & currentLvlMask) ? 1 : 0)
				+ ((iPos.y & currentLvlMask) ? 2 : 0)
				+ ((iPos.z & currentLvlMask) ? 4 : 0)
				+ offset;
			currVal = m_nodes[idx];
		}
		return currVal.get_sample();
	}

private:
	ei::Vec3 m_diagonalInv;
	ei::Vec3 m_minBound;
	const NodeType* m_nodes;
	const NodeType& m_root;
	u32 m_depth;
};

struct FloatOctreeNode {
	float data;

	// Creates a node that is pointing to other nodes as a parent
	CUDA_FUNCTION static FloatOctreeNode as_parent(const u32 offset) noexcept {
		return FloatOctreeNode{ -static_cast<float>(offset) };
	}
	CUDA_FUNCTION static FloatOctreeNode as_split_child(const float initViewCum) noexcept {
		return FloatOctreeNode{ initViewCum };
	}

	CUDA_FUNCTION bool is_parent() const noexcept { return data < 0.f; }
	CUDA_FUNCTION bool is_leaf() const noexcept { return !is_parent(); }

	// This function is purely for the spinlock functionality in case
	// the capacity limit has been reached
	CUDA_FUNCTION bool is_parent_or_fresh() const noexcept { return data <= 0.f; }

	CUDA_FUNCTION u32 get_child_offset() const noexcept {
		mAssert(this->is_parent());
		return static_cast<u32>(-data);
	}
	CUDA_FUNCTION float get_value() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
	CUDA_FUNCTION float get_sample() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
};

struct SampleOctreeNode {
	i32 count;
	float data;

	// Creates a node that is pointing to other nodes as a parent
	CUDA_FUNCTION static SampleOctreeNode as_parent(const u32 offset) noexcept {
		return SampleOctreeNode{ -static_cast<i32>(offset), 0.f };
	}
	CUDA_FUNCTION static SampleOctreeNode as_split_child(const u32 initCount, const float initValue) noexcept {
		return SampleOctreeNode{ static_cast<i32>(initCount), initValue };
	}

	CUDA_FUNCTION bool is_parent() const noexcept { return count < 0; }
	CUDA_FUNCTION bool is_leaf() const noexcept { return !is_parent(); }

	// This function is purely for the spinlock functionality in case
	// the capacity limit has been reached
	CUDA_FUNCTION bool is_parent_or_fresh() const noexcept { return count <= 0; }

	CUDA_FUNCTION u32 get_child_offset() const noexcept {
		mAssert(this->is_parent());
		return static_cast<u32>(-count);
	}
	CUDA_FUNCTION u32 get_count() const noexcept {
		mAssert(this->is_leaf());
		return count;
	}
	CUDA_FUNCTION float get_value() const noexcept {
		mAssert(this->is_leaf());
		return data;
	}
	CUDA_FUNCTION float get_sample() const noexcept {
		mAssert(this->is_leaf());
		return data / static_cast<float>(ei::max(1, count));
	}
};

class FloatOctree final : public Octree<FloatOctreeNode> {
public:
	using NodeType = FloatOctreeNode;

	__host__ FloatOctree(const ei::Box& bounds, u32 capacity, std::atomic<NodeType>* nodes,
						 std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter,
						 const float splitVal);
	FloatOctree(const FloatOctree&) = delete;
	__host__ FloatOctree(FloatOctree&& other) noexcept;
	FloatOctree& operator=(const FloatOctree&) = delete;
	FloatOctree& operator=(FloatOctree&&) = delete;
	virtual ~FloatOctree() = default;
	
	template < class O >
	__host__ void join(const Octree<O>& other, const float weight) noexcept;

protected:
	__host__ std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
												  const ei::UVec3& gridPos, const ei::Vec3& offPos,
												  const float value, const u32 depth) noexcept final;

	__host__ void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
								const ei::Vec3& offPos, const std::size_t depth,
								const NodeType& parentValue, const u32 childOffset) final;

private:
	float m_splitViewVal;
};


class SampleOctree final : public Octree<SampleOctreeNode> {
public:
	using NodeType = SampleOctreeNode;

	__host__ SampleOctree(const ei::Box& bounds, u32 capacity, std::atomic<NodeType>* nodes,
						  std::atomic<NodeType>& root, std::atomic_size_t& allocationCounter,
						  const u32 splitCount, const float splitVal);
	SampleOctree(const SampleOctree&) = delete;
	__host__ SampleOctree(SampleOctree&& other) noexcept;
	SampleOctree& operator=(const SampleOctree&) = delete;
	SampleOctree& operator=(SampleOctree&&) = delete;
	virtual ~SampleOctree() = default;

protected:
	__host__ std::atomic<NodeType>* add_and_split(std::atomic<NodeType>* curr, const ei::Vec3& normal,
												  const ei::UVec3& gridPos, const ei::Vec3& offPos,
												  const float value, const u32 depth) noexcept final;

	__host__ void init_children(const ei::Vec3& normal, const ei::UVec3& gridPos,
								const ei::Vec3& offPos, const std::size_t depth,
								const NodeType& parentValue, const u32 childOffset) final;

private:
	u32 m_splitCount;
	float m_splitValue;
};

}}} // namespace mufflon::renderer::decimaters