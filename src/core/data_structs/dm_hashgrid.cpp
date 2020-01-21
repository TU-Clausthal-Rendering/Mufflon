#include "dm_hashgrid.hpp"
#include <climits>

namespace mufflon::data_structs {

namespace {

template < class T >
void atomic_add(std::atomic<T>& atom, const T& value) {
	T expected = atom.load();
	T desired;
	do {
		desired = expected + value;
	} while(!atom.compare_exchange_weak(expected, desired));
}
template <>
void atomic_add<u32>(std::atomic<u32>& atom, const u32& value) {
	atom.fetch_add(value);
}

} // namespace

template < class T >
DmHashGrid<T>::DmHashGrid(const u32 numExpectedEntries) :
	m_cellSize{ 1.f },
	m_densityScale{ 1.f },
	m_mapSize{ ei::nextPrime(u32(numExpectedEntries * 1.15f)) },
	m_maxProbes{ m_mapSize / 2 },
	m_data{ std::make_unique<Entry[]>(m_mapSize) },
	m_dataCount{ 0u }
{}

template < class T >
DmHashGrid<T>::DmHashGrid(DmHashGrid&& grid) :
	m_cellSize(grid.m_cellSize),
	m_densityScale(grid.m_densityScale),
	m_mapSize(grid.m_mapSize),
	m_maxProbes(grid.m_maxProbes),
	m_data(std::move(grid.m_data)),
	m_dataCount(grid.m_dataCount.load())
{}

template < class T >
void DmHashGrid<T>::increase_count(const ei::Vec3& position, const T& value) {
	ei::UVec3 gridPos = get_grid_cell(position);
	u32 h = get_cell_hash(gridPos);
	u32 i = h % m_mapSize;
	int s = 1;
	// Quadratic probing until we find the correct or the empty cell
	while(s <= static_cast<int>(m_maxProbes)) {
		T expected{};
		// Check on empty and set a marker to allocate if empty
		if(m_data[i].count.compare_exchange_strong(expected, std::numeric_limits<T>::max())) {
			// The cell was empty before -> initialize
			m_data[i].cell = gridPos;
			m_data[i].count.store(value);	// Releases the lock at the same time as setting the correct count
			m_dataCount.fetch_add(1u);
			return;
		} else if(expected != std::numeric_limits<T>::max()) { // Not a cell marked as 'in allocation'
			if(m_data[i].cell == gridPos) {
				atomic_add(m_data[i].count, value);
				return;
			}
			// Next probe: non-empty cell with different coordinate found
			i = (h + (s & 1 ? s * s : -s * s) + m_mapSize) % m_mapSize;
			++s;
		} // else spin-lock (achieved by not changing i)
	}
}

template < class T >
float DmHashGrid<T>::get_density(const ei::Vec3& position, const ei::Vec3& normal) const {
	const ei::IVec3 gridPosI = ei::floor(position / m_cellSize);
	const ei::UVec3 gridPos{ gridPosI };
	const T c = get_count(gridPos);
	// Determine intersection area between cell and query plane
	ei::Vec3 localPos = position - gridPosI * m_cellSize;
	float area = math::intersection_area_nrm(m_cellSize, localPos, normal);
	return c * m_densityScale / area;
}

template < class T >
template < bool UseSmoothStep >
float DmHashGrid<T>::get_density_interpolated(const ei::Vec3& position, const ei::Vec3& normal, ei::Vec3* gradient) const {
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

template < class T >
T DmHashGrid<T>::get_count(const ei::UVec3& gridPos) const {
	u32 h = get_cell_hash(gridPos);
	u32 i = h % m_mapSize;
	int s = 1;
	T c = m_data[i].count.load();
	while(c > 0 && m_data[i].cell != gridPos) {
		i = (h + (s & 1 ? s * s : -s * s) + m_mapSize) % m_mapSize;
		++s;
		c = m_data[i].count.load();
	}
	return c;
}

template class DmHashGrid<i32>;
template class DmHashGrid<u32>;
template class DmHashGrid<float>;
template float DmHashGrid<i32>::get_density_interpolated<true>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid<i32>::get_density_interpolated<false>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid<u32>::get_density_interpolated<true>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid<u32>::get_density_interpolated<false>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid<float>::get_density_interpolated<true>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;
template float DmHashGrid<float>::get_density_interpolated<false>(const ei::Vec3&, const ei::Vec3&, ei::Vec3*) const;

} // namespace mufflon::data_structs