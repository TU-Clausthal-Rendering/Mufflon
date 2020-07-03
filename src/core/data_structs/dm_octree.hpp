#pragma once

#include "core/math/intersection_areas.hpp"
#include "util/log.hpp"
#include <ei/3dtypes.hpp>
#include <memory>
#include <atomic>

namespace mufflon { namespace data_structs {

// A sparse octree with atomic insertion to measure the density of elements in space.
template < class T = std::int32_t >
class DmOctree {
	// At some time the counting should stop -- otherwise the counter will overflow inevitable.
	static constexpr int FILL_ITERATIONS = 1000;
public:
	using DataType = T;

	// splitFactor: Number of photons in one cell (per iteration) before it is splitted.
	DmOctree(const ei::Box& sceneBounds, int capacity, float splitFactor, bool progressive);
	DmOctree(const DmOctree&) = delete;
	DmOctree(DmOctree&&);
	DmOctree& operator=(const DmOctree&) = delete;
	DmOctree& operator=(DmOctree&&) = default;
	~DmOctree() = default;

	// Initialize iteration count dependent data (1-indexed).
	// The first iteration must call set_iteration(1);
	void set_iteration(int iter);

	// Overwrite all counters with 0, but keep allocation and child pointers.
	void clear_counters();

	// Clear entire structure
	void clear();

	void increase_count(const ei::Vec3& pos, const ei::Vec3& normal, const DataType& value = DataType{ 1 });

	// Idea for truly smooth interpolation:
	//	Track the center positions of the cells to interpolate. Those might be
	//	the same for multiple coordinates. The shape formed by these centers is
	//	a convex polyhedron. Then one needs a kind of general barycentric coordinates
	//	to interpolate within this polyhedron:
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf (with pseudocode)
	//		* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.9919&rep=rep1&type=pdf
	//		* 2D: http://geometry.caltech.edu/pubs/MHBD02.pdf
	float get_density(const ei::Vec3& pos, const ei::Vec3& normal) const;

	float get_density_interpolated(const ei::Vec3& pos, const ei::Vec3& normal) const;

	void balance(int current = 0, int nx = 0, int ny = 0, int nz = 0, int px = 0, int py = 0, int pz = 0);

	// dir: 0 search in negative dir, 1 search in positive dir
	int find_neighbor(const ei::IVec3& localPos, int dim, int dir, int parentNeighbor, int siblings);

	int capacity() const { return m_capacity; }
	int size() const { return ei::min(m_capacity, m_allocationCounter.load()); }
	// Get the size of the associated memory excluding this instance.
	std::size_t mem_size() const { return sizeof(std::atomic_int32_t) * m_capacity; }
private:
	class NodeData {
	public:
		std::atomic<DataType>& operator*() noexcept {
			return m_value;
		}

		void operator=(DataType val) noexcept {
			m_value.store(val);
		}
		operator DataType() const noexcept {
			return m_value.load();
		}
	private:
		std::atomic<DataType> m_value{};
		//CritSection lock{};
		//std::mutex lock{};
	};


	float m_densityScale;		// 1/#iterations to normalize the counters into a density
	float m_splitFactor;		// Number of photons per iteration before a cell is split
	int m_splitCountDensity;	// Total number of photons before split
	float m_progression;		// Exponent of the progressive update function (0-maximum speed, 1-no progression)
	ei::Vec3 m_sceneSize;
	ei::Vec3 m_sceneSizeInv;
	ei::Vec3 m_minBound;
	// Nodes consist of 8 atomic counters OR child indices. Each number is either a
	// counter (positive) or a negated child index.
	int m_capacity;
	std::unique_ptr<NodeData[]> m_nodes;
	std::atomic_int32_t m_allocationCounter;
	std::atomic_int32_t m_depth;
	bool m_stopFilling;

	static constexpr ei::IVec3 CELL_ITER[8] = {
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
		{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
	};

	bool has_children(const DataType& val) const noexcept;
	int get_child_offset(const DataType& val) const noexcept;
	DataType create_child_offset(const int offset) const noexcept;

	// Returns the new value
	DataType increment_if_positive(int idx, const DataType& val);

	// Returns the new child pointer or 0
	DataType split_node_if_necessary(int idx, DataType value, int currentDepth,
									 const ei::IVec3& gridPos, const ei::Vec3& offPos,
									 const ei::Vec3& normal);

	// Non-atomic unconditional split. Returns the new address
	DataType split(int idx);

	// Set the counter of all unused cells to the number of expected samples
	// times 2. A planar surface will never extend to all eight cells. It might
	// intersect 7 of them, but still the distribution is one of a surface.
	// Therefore, the factor 2 (distribute among 4 cells) gives a much better initial
	// value.
	void init_children(int children, DataType /*count*/);

	void init_children(int children, DataType count, int currentDepth,
					   const ei::IVec3& gridPos, const ei::Vec3& offPos,
					   const ei::Vec3& normal);
};

}} // namespace mufflon::data_structs
