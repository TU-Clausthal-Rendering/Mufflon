#pragma once

#include "residency.hpp"
#include "core/export/core_api.h"
#include "generic_resource.hpp"
#include "util/types.hpp"
#include "util/assert.hpp"
#include <ei/prime.hpp>
#include <atomic>

// We need this since, even though it's marked as __device__,
// C++ requires a declaration
__device__ unsigned atomicInc(unsigned*, unsigned);
__device__ unsigned atomicCAS(unsigned*, unsigned, unsigned);

namespace mufflon {

template < typename K > CUDA_FUNCTION __forceinline__
u32 generic_hash(K key) {
	// Selfmade medium quality, but very fast hash (5 ops per 32bit word in the loop)
	// Passes the most important tests of SMHasher at sufficient quality
	u32 x = 0xa136aaadu;
	const u32* pkey = reinterpret_cast<u32*>(&key);
	for(int i = 0; i < static_cast<int>(sizeof(K) / 4); ++i, ++pkey) {
		x ^= *pkey * 0x11u;
		x = (x ^ (x >> 15)) * 0x469e0db1u;
	}
	// Get the last few bytes if there are some (should be removed as dead code if
	// sizeof(K) is a multiple of 4.
	if(sizeof(K) & 3) {
		// Make sure to read only the front-most n-bytes (n = sizeof%4 = sizeof&3)
		uint32_t k = *pkey & (0xffffffffull >> (32u - 8u * (sizeof(K) & 3u)));
		k += sizeof(K) & 3;
		x ^= k;
		x = (x ^ (x >> 17)) * 0x469e0db1u;
	}

	// Finalize using the best(?) known finalizer
	// https://github.com/skeeto/hash-prospector
	x ^= sizeof(K);		// Makes a hugh difference in some SMHasher tests.
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

/*
 * Parallel hash map, which can be used on CPU and CUDA.
 * This hash map does not implement a resizing mechanism. Be sure to create
 * a sufficiently large map from the beginning.
 *
 * When inserting items, the key must be unique. It is not allowed to insert
 * data with a key that is already contained in the map.
 * The reason is that data and mapping are inserted independent (parallel).
 * If there is a reason to overwrite values use the search function first.
 */
template < Device dev, typename K, typename V >
class HashMap;

template < typename K, typename V >
class HashMap<Device::CPU, K, V> {
	HashMap(u32 dataCapacity, u32 mapSize, char* map, char* data, std::atomic_uint32_t* counter) :
		m_data(as<std::pair<K,V>>(data)),
		m_map(as<std::atomic_uint32_t>(map)),
		m_dataCount(counter),
		m_mapSize(mapSize),
		m_dataCapacity(dataCapacity)
	{
		for(u32 i = 0u; i < m_mapSize; ++i)
			m_map[i].store(~0u);
	}
public:
	HashMap() : m_data(nullptr), m_map(nullptr), m_dataCount(nullptr), m_mapSize(0) {}
	HashMap(const HashMap&) = default;
	HashMap(HashMap&&) = default;
	HashMap& operator = (const HashMap&) = default;
	HashMap& operator = (HashMap&&) = default;

	// Insert a pair without checking, if the key is contained. See documentation above for deteails.
	void insert(K key, V value) {
		u32 hash = generic_hash(key);
		// Insert the datum
		u32 dataIdx = m_dataCount->fetch_add(1);
		mAssert(dataIdx < m_dataCapacity);
		m_data[dataIdx].first = key;
		m_data[dataIdx].second = value;
		// Try to insert until we find an empty entry
		u32 idx = hash % m_mapSize;
		u32 step = 0;
		u32 expected = ~0u;
		while(!std::atomic_compare_exchange_strong(&m_map[idx], &expected, dataIdx)) {
			mAssertMsg(!(m_data[expected].first == key), "Not allowed to add the same value twice.");
			++step;
			if(step & 1) idx = idx + step * step;
			else         idx = idx - step * step + m_mapSize;
			idx = idx % m_mapSize;
			expected = ~0u;
		}
	}

	// Finds an item or returns nullptr.
	V* find(K key) {
		u32 hash = generic_hash(key);
		u32 idx = hash % m_mapSize;
		u32 step = 0;
		u32 res;
		while((res = std::atomic_load(&m_map[idx])) != ~0u) {
			if(m_data[res].first == key) return &m_data[res].second;
			++step;
			if(step & 1) idx = idx + step * step;
			else         idx = idx - step * step + m_mapSize;
			idx = idx % m_mapSize;
		}
		return nullptr;
	}
	const V* find(K key) const { return const_cast<HashMap*>(this)->find(key); }

	// Get the number of elements in the hash map
	u32 size() const { return m_dataCount->load(); }
private:
	std::pair<K,V>* m_data;
	std::atomic_uint32_t* m_map;
	std::atomic_uint32_t* m_dataCount;
	u32 m_mapSize;
	u32 m_dataCapacity;

	template < typename K1, typename V1 >
	friend class HashMapManager;
};

template < typename K, typename V >
class HashMap<Device::CUDA, K, V> {
	HashMap(u32 /*dataCapacity*/, u32 mapSize, char* map, char* data, std::atomic_uint32_t* counter) :
		m_data(as<std::pair<K,V>>(data)),
		m_map(as<u32>(map)),
		m_mapSize(mapSize),
		m_dataCount(*counter)
	{}
public:
	HashMap() : m_data(nullptr), m_map(nullptr), m_mapSize(0), m_dataCount(0) {}
	HashMap(const HashMap&) = default;
	HashMap(HashMap&&) = default;
	HashMap& operator = (const HashMap&) = default;
	HashMap& operator = (HashMap&&) = default;
	// See CPU implementation for documentation

	__device__ void insert(K key, V value) {
		u32 hash = generic_hash(key);
		u32 dataIdx = atomicInc(&m_dataCount, 0);
		m_data[dataIdx].first = key;
		m_data[dataIdx].second = value;
		u32 idx = hash % m_mapSize;
		u32 step = 0;
		while(atomicCAS(&m_map[idx], ~0u, dataIdx) != ~0u) {
			++step;
			if(step & 1) idx = idx + step * step;
			else         idx = idx - step * step + m_mapSize;
			idx = idx % m_mapSize;
		}
	}

	__device__ V* find(K key) {
		u32 hash = generic_hash(key);
		u32 idx = hash % m_mapSize;
		u32 step = 0;
		u32 res;
		while((res = m_map[idx]) != ~0u) { // NO ATOMIC-LOAD ON CUDA??
			if(m_data[res].first == key) return &m_data[res].second;
			++step;
			if(step & 1) idx = idx + step * step;
			else         idx = idx - step * step + m_mapSize;
			idx = idx % m_mapSize;
		}
		return nullptr;
	}
	__device__ const V* find(K key) const { return const_cast<HashMap*>(this)->find(key); }

	// Get the number of elements in the hash map
	__device__ u32 size() const { return m_dataCount; }
private:
	std::pair<K,V>* m_data;
	u32* m_map;
	u32 m_mapSize;
	u32 m_dataCount;

	template < typename K1, typename V1 >
	friend class HashMapManager;
};

// TODO GL same as cpu right now
template < typename K, typename V >
class HashMap<Device::OPENGL, K, V> : public HashMap<Device::CPU, K, V> {};


/*
 * Management layer to support hashmap functionallity on all devices.
 */
template < typename K, typename V >
class HashMapManager {
	// Only primes congruent to 3 modulo 4 guarantee a well behaved quadratic probing
	u32 compute_valid_size(u32 desiredSize) {
		do {
			desiredSize = ei::nextPrime(desiredSize);
		} while((desiredSize & 3) != 3);
		return desiredSize;
	}
public:
	HashMapManager(int numExpectedEntries = 0) :
		m_cpuHMCounter{0}
	{
		resize(numExpectedEntries);
	}

	// Reallocates the map (previous data is lost)
	void resize(int numExpectedEntries) {
		m_dataCapacity = numExpectedEntries;
		m_mapSize = compute_valid_size(u32(numExpectedEntries * 1.15f));
		m_memory.resize(m_dataCapacity * sizeof(std::pair<K,V>) + m_mapSize * sizeof(u32));
	}

	template < Device dstDev >
	void synchronize() {
		m_memory.synchronize<dstDev>();
	}

	template< Device dev >
	void unload() {
		m_memory.unload<dev>();
	}

	void clear() {
		unload<Device::CPU>();
		unload<Device::CUDA>();
		unload<Device::OPENGL>();
		m_cpuHMCounter.store(0u);
	}

	// Get the functional HashMap
	template< Device dev >
	HashMap<dev,K,V> acquire() {
		char* ptr = m_memory.acquire<dev>();
		return HashMap<dev,K,V>{
			m_dataCapacity, m_mapSize,
			ptr, ptr + m_mapSize * sizeof(u32),
			&m_cpuHMCounter
		};
	}
	template< Device dev >
	const HashMap<dev,K,V> acquire_const() {
		return acquire<dev>();
	}

	void mark_changed(Device changed) noexcept {
		m_memory.mark_changed(changed);
	}
private:
	GenericResource m_memory;	// Contains both: hash table and data
	u32 m_mapSize;				// Size of the hash table
	u32 m_dataCapacity;			// Maximum number of data elements
	std::atomic_uint32_t m_cpuHMCounter;		// Store the atomic counter here, because the returned HashMap<CPU> is not trivially copyable otherwise
};

} // namespace mufflon
