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
	DmHashGrid(uint numExpectedEntries) {
		m_cellSize = ei::Vec3 { 1.0f };
		m_mapSize = ei::nextPrime(u32(numExpectedEntries * 1.15f));
		m_data.reset(new Entry[m_mapSize]);
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
			m_data[i].count.store(0u, std::memory_order_relaxed);
	}

	// Cell size is the major influence parameter for the performance.
	// The implementation always assumes that the query radius * 2 is less
	// than the cell size.
	void set_cell_size(float cellSize) { m_cellSize = ei::Vec3{ cellSize }; }
	float get_cell_size() const { return m_cellSize.x; }

	// Call in each iteration to make sure the density is scaled properly
	void set_iteration(int iter) { m_densityScale = 1.0f / iter; }

	// Increase the counter for a cell using a world position.
	void increase_count(const ei::Vec3& position) {
		ei::UVec3 gridPos = get_grid_cell(position);
		u32 h = get_cell_hash(gridPos);
		u32 i = h % m_mapSize;
		int s = 1;
		// Quadratic probing until we find the correct or the empty cell
		while(true) {
			u32 expected = 0u;
			// Check on empty and set a marker to allocate if empty
			if(m_data[i].count.compare_exchange_strong(expected, ~0u)) {
				// The cell was empty before -> initialize
				m_data[i].cell = gridPos;
				m_data[i].count.store(1u);	// Releases the lock at the same time as setting the correct count
				return;
			} else if(expected != ~0u) { // Not a cell marked as 'in allocation'
				if(m_data[i].cell == gridPos) {
					m_data[i].count.fetch_add(1);
					return;
				}
				// Next probe: non-empty cell with different coordinate found
				i = (h + (s&1 ? s*s : -s*s) + m_mapSize) % m_mapSize;
				++s;
			} // else spin-lock (achieved by not changing i)
		}
	}

	float get_density(const ei::Vec3& position, const ei::Vec3& normal) const {
		ei::IVec3 gridPosI = ei::floor(position / m_cellSize);
		ei::UVec3 gridPos { gridPosI };
		u32 c = get_count(gridPos);
		// Determine intersection area between cell and query plane
		ei::Vec3 localPos = position - gridPosI * m_cellSize;
		float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
		return c * m_densityScale / area;
	}

	float get_density_interpolated(const ei::Vec3& position, const ei::Vec3& normal) const {
		// Get integer and interpolation coordinates
		ei::Vec3 nrmPos = position / m_cellSize - 0.5f;
		ei::IVec3 gridPosI = ei::floor(nrmPos);
		ei::Vec3 ws[2];
		ws[1] = nrmPos - gridPosI;
		ws[0] = 1.0f - ws[1];
		float countSum = 0.0f, areaSum = 0.0f;
		// Iterate over all eight cells
		for(int i = 0u; i < 8u; ++i) {
			int ix = i & 1, iy = (i>>1) & 1, iz = i>>2;
			ei::IVec3 cellPos { gridPosI.x + ix,
								gridPosI.y + iy,
								gridPosI.z + iz };
			ei::Vec3 localPos = position - cellPos * m_cellSize;
			float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
			float count = static_cast<float>(get_count(ei::UVec3{cellPos}));
			// Compute trilinear interpolated result of count and area (running sum)
			float w = ws[ix].x * ws[iy].y * ws[iz].z;
			areaSum += area * w;
			countSum += count * w;
		}
		return sdiv(countSum, areaSum) * m_densityScale;
	}

private:
	struct Entry {
		ei::UVec3 cell;
		std::atomic_uint32_t count = 0;
	};

	ei::Vec3 m_cellSize;
	// The density scale can be used to if multiple iterations are accumulated
	// in the map. Set to 1/iterationCount to get correct densities.
	float m_densityScale;
	u32 m_mapSize;
	std::unique_ptr<Entry[]> m_data;

	u32 get_count(const ei::UVec3& gridPos) const {
		u32 h = get_cell_hash(gridPos);
		u32 i = h % m_mapSize;
		int s = 1;
		u32 c = m_data[i].count.load();
		while(c > 0 && m_data[i].cell != gridPos) {
			i = (h + (s&1 ? s*s : -s*s) + m_mapSize) % m_mapSize;
			++s;
			c = m_data[i].count.load();
		}
		return c;
	}
};

}} // namespace mufflon::data_structs
