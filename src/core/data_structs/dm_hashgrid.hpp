#pragma once

#include "core/math/intersection_areas.hpp"
#include <ei/vector.hpp>
#include <ei/prime.hpp>
#include <atomic>
#include <vector>

namespace mufflon { namespace data_structs {

// A lock-free atomic counter hash grid for fast density estimates.
class DmHashGrid {
public:
	// Create a hash grid with a fixed memory footprint. This hash grid does not
	// implement a resizing mechanism, so if you try to add more elements than
	// the expected number data is lost.
	DmHashGrid(uint numExpectedEntries, float cellSize)
	{
		m_cellSize = ei::Vec3 { cellSize };
		m_mapSize = ei::nextPrime(u32(numExpectedEntries * 1.15f));
		m_count.reset(new std::atomic_uint32_t[m_mapSize]);
		m_densityScale = 1.0f;
	}

	ei::UVec3 get_grid_cell(const ei::Vec3& position) const {
		return ei::UVec3(ei::floor(position / m_cellSize));
	}

	static u32 get_cell_hash(const ei::UVec3& cell) {
		// See core/renderer/photon_map.hpp for details
		constexpr ei::UVec3 MAGIC {0xb286aff7, 0x35e4a487, 0x75a9c18f};
		return dot(cell, MAGIC);
	}

	void clear() {
		for(u32 i = 0; i < m_mapSize; ++i)
			m_count[i].store(0u, std::memory_order_relaxed);
	}

	// Cell size is the major influence parameter for the performance.
	// The implementation always assumes that the query radius * 2 is less
	// than the cell size.
	void set_cell_size(float cellSize) { m_cellSize = ei::Vec3{ cellSize }; }
	float get_cell_size() const { return m_cellSize.x; }

	// The density scale can be used to if multiple iterations are accumulated
	// in the map. Set to 1/iterationCount to get correct densities.
	void set_density_scale(float scale) { m_densityScale = scale; }
	float get_density_scale() const { return m_densityScale; }

	// Increase the counter for a cell using a world position.
	u32 increase_count(const ei::Vec3& position) {
		u32 h = get_cell_hash(get_grid_cell(position));
		u32 i = h % m_mapSize;
		return m_count[i].fetch_add(1);
	}

	float get_density(const ei::Vec3& position, const ei::Vec3& normal) const {
		ei::IVec3 gridPos = ei::floor(position / m_cellSize);
		// Get count
		u32 h = get_cell_hash(ei::UVec3{gridPos});
		u32 i = h % m_mapSize;
		u32 c = m_count[i].load();
		// Determine intersection area between cell and query plane
		ei::Vec3 localPos = position - gridPos * m_cellSize;
		float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
		return c * m_densityScale / area;
	}

private:
	ei::Vec3 m_cellSize;
	float m_densityScale;
	u32 m_mapSize;
	std::unique_ptr<std::atomic_uint32_t[]> m_count;
};

}} // namespace mufflon::util::density
