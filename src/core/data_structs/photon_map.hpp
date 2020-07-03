#pragma once

#include "core/memory/residency.hpp"
#include "core/export/core_api.h"
#include "core/memory/allocator.hpp"
#include "core/memory/generic_resource.hpp"
#include "util/types.hpp"
#include "util/assert.hpp"
#include <ei/prime.hpp>
#include <atomic>

namespace mufflon { namespace data_structs {

/*
 * Hash-grid implementation. A hash grid is extremly fast and well parallelizable
 * if used with constant query radii.
 * In a hash grid the positions are descretized to a spaceing in the size of the query
 * diameter. Then, a query need to check 8 cells only to find all photons.
 *		Insertion: O(1) - one hash computation, two atomics
 *		Query: O(k) - for k photons in the 8 cells (~ number of results)
 */

// Device independent functionallity
class HashGridCommon {
protected:
	float m_cellDensity;		//< 1 / cellEdgeLength

	// 3 Magic numbers to scrample intput positions for the hash computation
	static constexpr ei::UVec3 MAGIC {0xb286aff7, 0x35e4a487, 0x75a9c18f};

	inline CUDA_FUNCTION ei::UVec3 get_grid_cell(const ei::Vec3& position) {
		return ei::UVec3(ei::ceil(position * m_cellDensity));
	}

	static inline CUDA_FUNCTION u32 get_cell_hash(const ei::UVec3& cell) {
		// Use the Cantor pairing function to compress ℕ³->ℕ
		//u32 preHash = ((cell.x + cell.y) * (cell.x + cell.y + 1)) / 2 + cell.y;
		//preHash = ((preHash + cell.z) * (preHash + cell.z + 1)) / 2 + cell.z;
		// Cantor pairing as a problem with the modulo property: if either x+y or x+y+1
		// get zero we get the same hash for two adjacent cells.
		// Szudzik pairing: same problem as cantor pairing.
		/*u32 preHash = cell.x > cell.y ? cell.x * cell.x + cell.y
									  : cell.y * cell.y + cell.y + cell.x;
		preHash = preHash > cell.z ? preHash * preHash + cell.z
								   : cell.z * cell.y + cell.z + preHash;*/
		//u32 preHash = cell.x * 0xb286aff7 + cell.y * 0x35e4a487 + cell.z * 0x75a9c18f;
		// Randomize it a bit (changes permutation) with Xorshift 32.
		/*preHash ^= preHash << 13;
		preHash ^= preHash >> 17;
		preHash ^= preHash << 5;*/
		return dot(cell, MAGIC);
	}

public:
	template < typename V >
	struct LinkedData {
		V data;
		u32 next;
	};
};

// The template which gets specialized for CPU and CUDA
template < Device dev, typename V >
class HashGrid;

// The CPU implementation
template < typename V >
class HashGrid<Device::CPU, V> : public HashGridCommon {
	HashGrid(u32 dataCapacity, u32 mapSize, char* map, char* data, std::atomic_uint32_t* counter) :
		m_map(as<std::atomic_uint32_t>(map)),
		m_dataCount(counter),
		m_data(as<LinkedData<V>>(data)),
		m_dataCapacity(dataCapacity),
		m_mapSize(mapSize)
	{
		clear(1.0f);
	}
public:
	HashGrid() :
		m_map{nullptr},
		m_dataCount{nullptr},
		m_data{nullptr},
		m_dataCapacity{0},
		m_mapSize{0}
	{}

	// Clear and reset the cell size. The cell size must be at least query radius * 2.
	void clear(float cellEdgeLength) {
		m_cellDensity = 1.0f / cellEdgeLength;
		m_dataCount->store(0, std::memory_order_relaxed);
		for(u32 i = 0; i < m_mapSize; ++i) {
			m_map[i].store(~0u, std::memory_order_relaxed);
		}
	}

	// Inserts a datum and returns its new address within the map. This address
	// will not change until clear().
	V* insert(const ei::Vec3& position, const V& _data)
	{
		u32 h = get_cell_hash(get_grid_cell(position));
		u32 i = h % m_mapSize;
		u32 datIdx = m_dataCount->fetch_add(1);
		if(datIdx >= m_dataCapacity) {
			mAssert(false);
			return nullptr; // LOST DATA DUE TO OVERFLOW!
		}
		u32 prevDatIdx = m_map[i].exchange(datIdx);
		m_data[datIdx].next = prevDatIdx;
		m_data[datIdx].data = _data;
		return &m_data[datIdx].data;
	}

	// Get the number of elements in the hash map
	u32 size() const { return ei::min(m_dataCapacity, m_dataCount->load()); }
	u32 capacity() const { return m_dataCapacity; }

	class NeighborIterator
	{
		const HashGrid<Device::CPU, V>* m_container;
		u32 m_baseHash;			//< Hash of the current cell with smallest index of all cells in a 8 neighborhood.
		//ei::UVec3 m_baseCell;	//< Smallest index of all cells in a 8 neighborhood.
		i32 m_cellIdx;			//< In [0,7] specifying one of the 8 neighbors.
		u32 m_datIdx;
		friend class HashGrid;
	public:
		NeighborIterator& operator ++ () {
			// Other data in the same cell?
			if(m_datIdx != ~0u && m_container->m_data[m_datIdx].next != ~0u)
				m_datIdx = m_container->m_data[m_datIdx].next;
			else {
				m_datIdx = ~0u;
				// Other cells to check?
				while(m_datIdx == ~0u && ++m_cellIdx < 8) {
					u32 h = m_baseHash;
					if(m_cellIdx & 1) h += MAGIC.x;
					if(m_cellIdx & 2) h += MAGIC.y;
					if(m_cellIdx & 4) h += MAGIC.z;
					u32 i = h % m_container->m_mapSize;
					m_datIdx = m_container->m_map[i];
				}
			}
			return *this;
		}

		const V& operator * () const { return m_container->m_data[m_datIdx].data; }
		const V* operator -> () const { return &m_container->m_data[m_datIdx].data; }
		bool operator == (const NeighborIterator& _other) const { return m_container == _other.m_container && m_datIdx == _other.m_datIdx; }
		bool operator != (const NeighborIterator& _other) const { return m_container != _other.m_container || m_datIdx != _other.m_datIdx; }
		operator bool () const { return m_datIdx != ~0u; }
	};

	// Gets an iterator over 8 cells assuming that a query at _position is made with
	// a diameter smaller then the grid spacing.
	NeighborIterator find_first(const ei::Vec3& position) const {
		NeighborIterator it;
		it.m_container = this;
		//it.m_baseCell = ei::UVec3(ei::round(position * m_cellDensity));
		ei::UVec3 baseCell = ei::UVec3(ei::round(position * m_cellDensity));
		it.m_baseHash = get_cell_hash(baseCell);
		it.m_cellIdx = -1;
		it.m_datIdx = ~0u;
		++it;
		return it;
	}

	/*class MapIterator {
		const HashGrid<Device::CPU, V>* m_container;
		u32 m_dataIdx = 0;
		u32 m_dataCount;
		friend class HashGrid;
	public:
		MapIterator& operator ++ () {
			++m_dataIdx;
			return *this;
		}

		const V& operator * () const { return m_container->m_data[m_dataIdx].data; }
		const V* operator -> () const { return &m_container->m_data[m_dataIdx].data; }
		bool operator == (const NeighborIterator& _other) const { return m_container == _other.m_container && m_dataIdx == _other.m_dataIdx; }
		bool operator != (const NeighborIterator& _other) const { return m_container != _other.m_container || m_dataIdx != _other.m_dataIdx; }
		operator bool () const { return m_dataIdx < m_dataCount; }
	};
	MapIterator begin() const {
		MapIterator it;
		it.m_container = this;
		it.m_dataCount = m_dataCount->load();
		return it;
	}
	MapIterator end() const {
		MapIterator it;
		it.m_container = this;
		it.m_dataCount = m_dataCount->load();
		it.m_dataIdx = it.m_dataCount;
		return it;
	}*/

	V& get_data_by_index(u32 index) { return m_data[index].data; }
	const V& get_data_by_index(u32 index) const { return m_data[index].data; }

private:
	std::atomic_uint32_t* m_map;
	std::atomic_uint32_t* m_dataCount;
	LinkedData<V>* m_data;
	u32 m_dataCapacity;
	u32 m_mapSize;

	template < typename V1 >
	friend class HashGridManager;
};

// The GPU implementation
template < typename V >
class HashGrid<Device::CUDA, V> : public HashGridCommon {
	// TODO
};


/*
 * Management layer to support hash grid functionallity on all devices.
 */
template < typename V >
class HashGridManager {
	// Only primes congruent to 3 modulo 4 guarantee a well behaved quadratic probing
	u32 compute_valid_size(u32 desiredSize) {
		do {
			desiredSize = ei::nextPrime(desiredSize);
		} while((desiredSize & 3) != 3);
		return desiredSize;
	}
public:
	HashGridManager(int numExpectedEntries = 0) :
		m_dataCapacity{ 0 },
		m_cpuHMCounter{ 0 }
	{
		resize(numExpectedEntries);
	}

	// Reallocates the map (previous data is lost)
	void resize(int numExpectedEntries) {
		if(m_dataCapacity != static_cast<u32>(numExpectedEntries)) {
			m_dataCapacity = numExpectedEntries;
			m_mapSize = compute_valid_size(u32(numExpectedEntries * 1.15f));
			m_memory.resize(m_dataCapacity * sizeof(HashGridCommon::LinkedData<V>) + m_mapSize * sizeof(u32));
		}
	}

	template < Device dstDev >
	void synchronize() {
		m_memory.synchronize<dstDev>();
	}

	template< Device dev >
	void unload() {
		m_memory.unload<dev>();
	}

	// Get the functional HashMap
	template< Device dev >
	HashGrid<dev,V> acquire() {
		char* ptr = m_memory.acquire<dev>();
		return HashGrid<dev,V>{
			m_dataCapacity, m_mapSize,
			ptr, ptr + m_mapSize * sizeof(u32),
			&m_cpuHMCounter
		};
	}
	template< Device dev >
	const HashGrid<dev,V> acquire_const() {
		return acquire<dev>();
	}

	void mark_changed(Device changed) noexcept {
		m_memory.mark_changed(changed);
	}

	// Get the size of assoziated memory blocks (does not count the header information
	// in the current instance).
	std::size_t mem_size() const { return m_memory.size(); }
private:
	GenericResource m_memory;	// Contains both: hash table and data
	u32 m_mapSize;				// Size of the hash table
	u32 m_dataCapacity;			// Maximum number of data elements
	std::atomic_uint32_t m_cpuHMCounter;		// Store the atomic counter here, because the returned HashMap<CPU> is not trivially copyable otherwise
};

} // namespace data_structs

template struct DeviceManagerConcept<data_structs::HashGridManager<int>>;

} // namespace mufflon