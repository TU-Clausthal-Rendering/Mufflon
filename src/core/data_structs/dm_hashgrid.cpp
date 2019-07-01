#include "dm_hashgrid.hpp"

namespace mufflon::data_structs {

DmHashGrid::DmHashGrid(const u32 numExpectedEntries) :
	m_cellSize{ 1.f },
	m_densityScale{ 1.f },
	m_mapSize{ ei::nextPrime(u32(numExpectedEntries * 1.15f)) },
	m_data{ std::make_unique<Entry[]>(m_mapSize) }
{}

void DmHashGrid::increase_count(const ei::Vec3& position) {
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
			i = (h + (s & 1 ? s * s : -s * s) + m_mapSize) % m_mapSize;
			++s;
		} // else spin-lock (achieved by not changing i)
	}
}
float DmHashGrid::get_density(const ei::Vec3& position, const ei::Vec3& normal) const {
	ei::IVec3 gridPosI = ei::floor(position / m_cellSize);
	ei::UVec3 gridPos{ gridPosI };
	u32 c = get_count(gridPos);
	// Determine intersection area between cell and query plane
	ei::Vec3 localPos = position - gridPosI * m_cellSize;
	float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
	return c * m_densityScale / area;
}

template < bool UseSmoothStep >
float DmHashGrid::get_density_interpolated(const ei::Vec3& position, const ei::Vec3& normal, ei::Vec3* gradient) const {
	// Get integer and interpolation coordinates
	ei::Vec3 nrmPos = position / m_cellSize - 0.5f;
	ei::IVec3 gridPosI = ei::floor(nrmPos);
	ei::Vec3 ws[2];
	ws[1] = nrmPos - gridPosI;
	ws[0] = 1.0f - ws[1];
	float countSum = 0.0f, areaSum = 0.0f;
	// Iterate over all eight cells
	for(int i = 0u; i < 8u; ++i) {
		int ix = i & 1, iy = (i >> 1) & 1, iz = i >> 2;
		ei::IVec3 cellPos{ gridPosI.x + ix,
							gridPosI.y + iy,
							gridPosI.z + iz };
		ei::Vec3 localPos = position - cellPos * m_cellSize;
		float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
		float count = static_cast<float>(get_count(ei::UVec3{ cellPos }));
		// Compute trilinear interpolated result of count and area (running sum)
		const float w = UseSmoothStep ? ei::smoothstep(ws[ix].x) * ei::smoothstep(ws[iy].y) * ei::smoothstep(ws[iz].z)
									  : ws[ix].x * ws[iy].y * ws[iz].z;
		countSum += count * w;
		areaSum += area * w;

		if(gradient != nullptr) {
			if constexpr(UseSmoothStep) {
				// Derivative for smooth step
				*gradient += ei::Vec3{
					(ix ? -1.f : 1.f) * 6.f * ws[1].x * ws[0].x * ws[iy].y * ws[iz].z * count,
					(iy ? -1.f : 1.f) * 6.f * ws[1].y * ws[0].y * ws[iy].x * ws[iz].z * count,
					(iz ? -1.f : 1.f) * 6.f * ws[1].z * ws[0].z * ws[iy].x * ws[iz].y * count,
				};
			} else {
				// Gradient for trilinear interpolation
				*gradient += ei::Vec3{
					(ix ? -1.f : 1.f) * ws[iy].y * ws[iz].z * count,
					(iy ? -1.f : 1.f) * ws[ix].x * ws[iz].z * count,
					(iz ? -1.f : 1.f) * ws[ix].x * ws[iy].y * count
				};
			}
		}
	}

	if(gradient != nullptr)
		*gradient = sdiv(*gradient, areaSum);

	return sdiv(countSum, areaSum) * m_densityScale;
}

u32 DmHashGrid::get_count(const ei::UVec3& gridPos) const {
	u32 h = get_cell_hash(gridPos);
	u32 i = h % m_mapSize;
	int s = 1;
	u32 c = m_data[i].count.load();
	while(c > 0 && m_data[i].cell != gridPos) {
		i = (h + (s & 1 ? s * s : -s * s) + m_mapSize) % m_mapSize;
		++s;
		c = m_data[i].count.load();
	}
	return c;
}

template float DmHashGrid::get_density_interpolated<true>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid::get_density_interpolated<false>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;

} // namespace mufflon::data_structs