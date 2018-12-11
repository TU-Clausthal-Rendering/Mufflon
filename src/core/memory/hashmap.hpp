#pragma once

#include "residency.hpp"
#include "core/export/api.h"
#include "allocator.hpp"
#include "util/types.hpp"
#include "util/assert.hpp"
#include <ei/prime.hpp>
#include <atomic>

namespace mufflon {

template < typename K > CUDA_FUNCTION __forceinline__
u32 generic_hash(K key) {
	// Selfmade medium quality, but very fast hash (5 ops per 32bit word in the loop)
	// Passes the most important tests of SMHasher at sufficient quality
	u32 x = 0xa136aaadu;
	const u32* pkey = reinterpret_cast<u32*>(&key);
	for(int i = 0; i < sizeof(K) / 4; ++i, ++pkey) {
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
public:
	HashMap(int numExpectedEntries) {
		m_dataSize = numExpectedEntries;
		m_mapSize = ei::nextPrime(u32(numExpectedEntries * 1.15f));
		m_memSize = numExpectedEntries * sizeof(std::pair<K,V>) + m_mapSize * sizeof(u32);
		m_memory.reset(new char[m_memSize]);
		m_data = as<std::pair<K,V>>(m_memory.get() + m_mapSize * sizeof(u32));
		m_map = as<std::atomic_uint32_t>(m_memory.get());
		m_dataCount.store(0);
	}

	// Insert a pair without checking, if the key is contained. See documentation above for deteails.
	void insert(K key, V value) {
		u32 hash = generic_hash(key);
		// Insert the datum
		u32 dataIdx = m_dataCount.fetch_add(1);
		mAssert(dataIdx < m_dataSize);
		m_data[dataIdx].first = key;
		m_data[dataIdx].second = value;
		// Try to insert until we find an empty entry
		u32 idx = hash % m_mapSize;
		u32 step = 0;
		u32 expected = ~0u;
		while(!std::atomic_compare_exchange_strong(&m_map[idx], &expected, dataIdx)) {
			mAssertMsg(m_data[expected].first != key, "Not allowed to add the same value twice.");
			++step;
			idx = (idx + step * step) % m_mapSize;
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
			idx = (idx + step * step) % m_mapSize;
		}
		return nullptr;
	}
	const V* find(K key) const { return const_cast<HashMap*>(this)->find(key); }

	// Get the number of elements in the hash map
	u32 size() const { return m_dataCount; }
private:
	std::unique_ptr<char[]> m_memory;
	std::pair<K,V>* m_data;
	std::atomic_uint32_t* m_map;
	u32 m_mapSize;
	u32 m_dataSize;
	u32 m_memSize;
	std::atomic_uint32_t m_dataCount;

	friend HashMap<Device::CUDA, K, V>;
};

template < typename K, typename V >
class HashMap<Device::CUDA, K, V> {
public:
	HashMap(int numExpectedEntries) noexcept {
		m_mapSize = ei::nextPrime(u32(numExpectedEntries * 1.15f));
		m_memSize = numExpectedEntries * sizeof(std::pair<K,V>) + m_mapSize * sizeof(u32);
		m_memory = Allocator<Device::CUDA>::alloc_array<char>(m_memSize);
		m_data = as<std::pair<K,V>>(m_memory + m_mapSize * sizeof(u32));
		m_map = as<u32>(m_memory);
		m_dataCount = 0;
	}

	void free() noexcept {
		Allocator<Device::CUDA>::free(m_memory, m_memSize);
	}
	// Creepy: CUDA requires the copy to push the data to the GPU.
	// However, any copy should never call ~HashMap();
	//HashMap(const HashMap&) = delete;
	//HashMap& operator=(const HashMap&) = delete;
	//HashMap(HashMap&&) = delete;
	//HashMap& operator=(HashMap&&) = delete;

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
			idx = (idx + step * step) % m_mapSize;
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
			idx = (idx + step * step) % m_mapSize;
		}
		return nullptr;
	}
	__device__ const V* find(K key) const { return const_cast<HashMap*>(this)->find(); }

	// Get the number of elements in the hash map
	__host__ __device__ u32 size() const { return m_dataCount; }

	void synchornize(const HashMap<Device::CPU, K, V>& other) {
		mAssert(m_memSize == other.m_memSize);
		m_dataCount = other.m_dataCount;
		cudaMemcpy(m_memory, other.m_memory.get(), m_memSize, cudaMemcpyDefault);
	}
private:
	ArrayDevHandle_t<Device::CUDA, char> m_memory;
	std::pair<K,V>* m_data;
	u32* m_map;
	u32 m_memSize;
	u32 m_mapSize;
	u32 m_dataCount;
};

} // namespace mufflon
