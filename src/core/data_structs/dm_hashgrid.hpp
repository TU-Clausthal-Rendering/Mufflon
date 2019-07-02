#pragma once

#include "core/math/intersection_areas.hpp"
#include "util/types.hpp"
#include <ei/vector.hpp>
#include <ei/prime.hpp>
#include <atomic>
#include <memory>
#include <vector>

namespace mufflon { namespace data_structs {

// A lock-free atomic counter hash grid for fast density estimates.
class DmHashGrid {
public:
	// Create a hash grid with a fixed memory footprint. This hash grid does not
	// implement a resizing mechanism, so if you try to add more elements than
	// the expected number data is lost.
	DmHashGrid(const u32 numExpectedEntries);
	DmHashGrid(const DmHashGrid&) = delete;
	DmHashGrid(DmHashGrid&&) = default;
	DmHashGrid& operator=(const DmHashGrid&) = delete;
	DmHashGrid& operator=(DmHashGrid&&) = delete;
	~DmHashGrid() = default;

	ei::UVec3 get_grid_cell(const ei::Vec3& position) const {
		return ei::UVec3(ei::floor(position / m_cellSize));
	}

	static constexpr u32 get_cell_hash(const ei::UVec3& cell) {
		// See core/renderer/photon_map.hpp for details
		constexpr ei::UVec3 MAGIC {0xb286aff7, 0x35e4a487, 0x75a9c18f};
		return dot(cell, MAGIC);
	}

	void clear() {
		for(u32 i = 0; i < m_mapSize; ++i)
			m_data[i].count.store(0, std::memory_order_relaxed);
	}

	// Cell size is the major influence parameter for the performance.
	// The implementation always assumes that the query radius * 2 is less
	// than the cell size.
	void set_cell_size(float cellSize) { m_cellSize = ei::Vec3{ cellSize }; }
	float get_cell_size() const { return m_cellSize.x; }

	// Call in each iteration to make sure the density is scaled properly
	void set_iteration(int iter) { m_densityScale = 1.0f / iter; }

	// Increase the counter for a cell using a world position.
	void increase_count(const ei::Vec3& position);

	// Returns the point-sampled density at the given position wrt. the area of the plane passing through the cell
	float get_density(const ei::Vec3& position, const ei::Vec3& normal) const;

	// Returns the linearly interpolated density at the given position wrt. the area of the plane passing through the cell.
	// Optionally returns the gradient of the density in each direction as well.
	// Optionally uses smoothstep instead of linear interpolation.
	template < bool UseSmoothStep = false >
	float get_density_interpolated(const ei::Vec3& position, const ei::Vec3& normal, ei::Vec3* gradient = nullptr) const;

private:
	struct Entry {
		ei::UVec3 cell;
		std::atomic_uint32_t count = 0;
	};

	u32 get_count(const ei::UVec3& gridPos) const;

	ei::Vec3 m_cellSize;
	// The density scale can be used to if multiple iterations are accumulated
	// in the map. Set to 1/iterationCount to get correct densities.
	float m_densityScale;
	const u32 m_mapSize;
	std::unique_ptr<Entry[]> m_data;
};

}} // namespace mufflon::data_structs
