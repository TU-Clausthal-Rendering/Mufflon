#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include <ei/prime.hpp>
#include <cstddef>
#include <functional>
#include <tuple>
#include <vector>

namespace mufflon::util {

/* This hashmap works on a fixed number of (maximum) entries.
 * In return it guarantees that no reallocation happens and
 * returned references stay valid until destruction.
 * Beware that this hashmap is NOT thread-safe!
 */
template < class Key, class Value >
class FixedHashMap final {
public:
	using key_type = Key;
	using mapped_type = Value;
	using value_type = std::pair<const key_type, mapped_type>;
	using size_type = std::size_t;
	using hasher = std::hash<key_type>;
	using reference = mapped_type&;
	using const_reference = const mapped_type&;
	using pointer = mapped_type*;
	using const_pointer = const mapped_type*;
	using iterator = typename std::vector<value_type>::iterator;
	using const_iterator = typename std::vector<value_type>::const_iterator;

	FixedHashMap() = default;
	FixedHashMap(const std::size_t entries) :
		m_capacity{ entries }
	{
		// Make sure we have a map size that allows for good collision resistance
		auto mapEntries = entries;
		do {
			mapEntries = ei::nextPrime(mapEntries);
		} while((mapEntries & 3) != 3);
		m_map.resize(mapEntries);
		m_data.reserve(entries);
		// Fill the map with indicator value
		std::fill(m_map.begin(), m_map.end(), ~0u);
	}
	FixedHashMap(const FixedHashMap&) = default;
	FixedHashMap(FixedHashMap&&) = default;
	FixedHashMap& operator=(const FixedHashMap&) = default;
	FixedHashMap& operator=(FixedHashMap&&) = default;
	~FixedHashMap() = default;

	// Insert a pair without checking if the key is contained
	reference insert(const key_type& key, mapped_type&& value) {
		if(size() >= max_size())
			throw std::runtime_error("Hashmap overflow!");
		// Insert the datum
		const auto dataIndex = m_data.size();
		m_data.emplace_back(std::move(key), std::move(value));
		add_map_entry(m_data.back().first, dataIndex);
		return m_data.back().second;
	}
	reference insert(const key_type& key, const mapped_type& value) {
		if(size() >= max_size())
			throw std::runtime_error("Hashmap overflow!");
		// Insert the datum
		const auto dataIndex = m_data.size();
		m_data.emplace_back(std::move(key), std::move(value));
		add_map_entry(m_data.back().first, dataIndex);
		return m_data.back().second;
	}

	template < class... Args >
	reference emplace(Args&& ...args) {
		if(size() >= max_size())
			throw std::runtime_error("Hashmap overflow!");
		// Insert the datum
		const auto dataIndex = m_data.size();
		m_data.emplace_back(std::forward<Args>(args)...);
		add_map_entry(m_data.back().first, dataIndex);
		return m_data.back().second;
	}

	reference at(const key_type& key) {
		if(auto iter = find(key); iter == m_data.end())
			throw std::out_of_range("Key not in hashmap");
		else
			return iter->second;
	}
	const_reference at(const key_type& key) const {
		if(auto iter = find(key); iter == m_data.end())
			throw std::out_of_range("Key not in hashmap");
		else
			return iter->second;
	}

	reference operator[](const key_type& key) {
		auto iter = find(key);
		if(iter == m_data.end())
			return insert(key, {});
		return iter->second;
	}

	iterator begin() noexcept { return m_data.begin(); }
	const_iterator begin() const noexcept { return m_data.begin(); }
	const_iterator cbegin() const noexcept { return m_data.cbegin(); }
	iterator end() noexcept { return m_data.end(); }
	const_iterator end() const noexcept { return m_data.end(); }
	const_iterator cend() const noexcept { return m_data.cend(); }

	iterator find(const key_type& key) noexcept {
		if(empty())
			return m_data.end();
		const std::size_t mapSize = m_map.size();
		const std::size_t hash = std::hash<key_type>()(key);
		std::size_t idx = hash % mapSize;
		std::size_t step = 0;
		std::size_t res;
		while((res = m_map[idx]) != ~0u) {
			if(m_data[res].first == key)
				return m_data.begin() + res;
			++step;
			if(step & 1)
				idx = idx + step * step;
			else
				idx = idx - step * step + mapSize;
			idx = idx % mapSize;
		}
		return m_data.end();
	}
	const_iterator find(const key_type& key) const noexcept {
		if(empty())
			return m_data.cend();
		const std::size_t mapSize = m_map.size();
		const std::size_t hash = std::hash<key_type>()(key);
		std::size_t idx = hash % mapSize;
		std::size_t step = 0;
		std::size_t res;
		while((res = m_map[idx]) != ~0u) {
			if(m_data[res].first == key)
				return m_data.begin() + res;
			++step;
			if(step & 1)
				idx = idx + step * step;
			else
				idx = idx - step * step + mapSize;
			idx = idx % mapSize;
		}
		return m_data.end();
	}

	std::size_t size() const noexcept { return m_data.size(); }
	bool empty() const noexcept { return size() == 0u; }
	std::size_t max_size() const noexcept { return m_capacity;}

private:
	void add_map_entry(const key_type& key, const std::size_t dataIndex) {
		const std::size_t mapSize = m_map.size();
		// Try to insert until we find an empty entry
		const std::size_t hash = hasher()(key);
		std::size_t idx = hash % mapSize;
		std::size_t step = 0;
		std::size_t expected = ~0u;
		while(m_map[idx] != expected) {
			mAssertMsg(m_data[m_map[idx]].first != key, "Not allowed to add the same value twice.");
			++step;
			if(step & 1)
				idx = idx + step * step;
			else
				idx = idx - step * step + mapSize;
			idx = idx % mapSize;
			expected = ~0u;
		}
		m_map[idx] = dataIndex;
	}

	std::vector<value_type> m_data;
	std::vector<std::size_t> m_map;
	std::size_t m_capacity;
};

} // namespace mufflon::util