#pragma once

#include "core/math/intersection_areas.hpp"
#include "util/log.hpp"
#include <ei/3dtypes.hpp>
#include <memory>
#include <atomic>

namespace mufflon { namespace data_structs {

// A sparse octree with atomic insertion to measure the density of elements in space.
template < class T = i32 >
class DmOctree {
	// At some time the counting should stop -- otherwise the counter will overflow inevitable.
	static constexpr int FILL_ITERATIONS = 1000;
public:
	using CountType = T;

	// splitFactor: Number of photons in one cell (per iteration) before it is splitted.
	DmOctree(const ei::Box& sceneBounds, const int capacity, const float splitFactor);
	DmOctree(const DmOctree&) = delete;
	DmOctree(DmOctree&&);
	DmOctree& operator=(const DmOctree&) = delete;
	DmOctree& operator=(DmOctree&&) = delete;
	~DmOctree() = default;

	// Initialize iteration count dependent data (1-indexed).
	// The first iteration must call set_iteration(1);
	void set_iteration(const int iter);

	// Overwrite all counters with 0, but keep allocation and child pointers.
	void clear_counters();

	// Clear entire structure
	void clear();

	void increase_count(const ei::Vec3& pos, const T value = T{ 1 });

	template < class V = float >
	V get_density(const ei::Vec3& pos, const ei::Vec3& normal) const;

	// Returns the linearly (optionally smoothstep) interpolated density and optionally its gradient.
	// Idea for truly smooth interpolation:
	//	Track the center positions of the cells to interpolate. Those might be
	//	the same for multiple coordinates. The shape formed by these centers is
	//	a convex polyhedron. Then one needs a kind of general barycentric coordinates
	//	to interpolate within this polyhedron:
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf (with pseudocode)
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.9919&rep=rep1&type=pdf
	//		* 2D: http://geometry.caltech.edu/pubs/MHBD02.pdf
	template < bool UseSmoothStep = false, class V = float >
	V get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal, ei::Vec3* gradient = nullptr) const;

	void balance(const int current = 0, const int nx = 0, const int ny = 0, const int nz = 0,
				 const int px = 0, const int py = 0, const int pz = 0);

	// dir: 0 search in negative dir, 1 search in positive dir
	int find_neighbor(const ei::IVec3& localPos, const int dim, const int dir,
					  const int parentNeighbor, const int siblings);

	int capacity() const { return m_capacity; }
	int size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
	// Get the size of the associated memory excluding this instance.
	std::size_t mem_size() const { return sizeof(std::atomic_int32_t) * m_capacity; }
private:
	float m_densityScale;		// 1/#iterations to normalize the counters into a density
	const float m_splitFactor;	// Number of photons per iteration before a cell is split
	int m_splitCountDensity;	// Total number of photons before split
	const ei::Vec3 m_sceneSize;
	const ei::Vec3 m_sceneSizeInv;
	const ei::Vec3 m_minBound;
	const int m_capacity;
	// Nodes consist of 8 atomic counters OR child indices. Each number is either a
	// counter (positive) or a negated child index.
	std::unique_ptr<std::atomic<CountType>[]> m_nodes;
	std::atomic_int32_t m_allocationCounter;
	std::atomic_int32_t m_depth;
	bool m_stopFilling;

	static constexpr bool is_child_pointer(const T value) {
		return value < T{ 0 };
	}

	static constexpr T mark_child_pointer(const T value) {
		return -value;
	}

	// Returns the new child pointer or 0
	T increment_if_child_and_split_if_necessary(const int idx, const T value, const int currentDepth);

	// Non-atomic unconditional split. Returns the new address
	T split(const int idx);
};

}} // namespace mufflon::data_structs
