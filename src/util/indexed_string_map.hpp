#pragma once

#include "types.hpp"

#include <vector>
#include <unordered_map>
#include <string>
#include "util/string_view.hpp"
#include <memory>

namespace mufflon { namespace util {
/*
 * The indexed string map solves two problems with strings:
 * 1. unordered_map does not support string_view lookups if storing string as key.
 *    (Key type must be the same in queries as in the definition of the map.)
 * 2. Constant time indexed acces.
 *    We need this for cross-C-interface iteration.
 */
template < typename DataT >
class IndexedStringMap {
public:
	u32 insert(std::string key, DataT&& value) {
		m_mapKeyStore.push_back(std::make_unique<std::string>(move(key)));
		m_data.push_back(std::move(value));
		(void)m_map.emplace(*m_mapKeyStore.back(), m_data.size()-1);
		return u32(m_data.size()-1);
	}

	void reserve(std::size_t n) {
		m_map.reserve(n);
		m_mapKeyStore.reserve(n);
		m_data.reserve(n);
	}

	void erase(std::size_t index) {
		m_map.erase(*m_mapKeyStore.at(index));
		m_mapKeyStore.erase(m_mapKeyStore.begin() + index);
		m_data.erase(m_data.begin() + index);
	}

	DataT& get(std::size_t index) { return m_data.at(index); }
	const DataT& get(std::size_t index) const { return m_data.at(index); }
	DataT* find(StringView name) {
		auto it = m_map.find(name);
		if(it == m_map.end()) return nullptr;
		return &m_data.at(it->second);
	}
	const DataT* find(StringView name) const {
		return const_cast<IndexedStringMap*>(this)->find(name);
	}

	const StringView get_key(std::size_t index) const {
		return *m_mapKeyStore.at(index);
	}
	const std::size_t get_index(StringView name) const {
		return m_map.at(name);
	}

	std::size_t size() const noexcept {
		return m_data.size();
	}

	bool empty() const noexcept {
		return size() == 0u;
	}

	// Replace the name of an item while retaining its index.
	void change_key(std::size_t index, std::string name) {
		m_map.erase(*m_mapKeyStore.at(index));
		m_mapKeyStore[index] = std::make_unique<std::string>(move(name));
		m_map.emplace(*m_mapKeyStore[index], index);
	}

	void clear() {
		m_map.clear();
		m_mapKeyStore.clear();
		m_data.clear();
	}

private:
	std::unordered_map<StringView, std::size_t> m_map;
	std::vector<std::unique_ptr<std::string>> m_mapKeyStore;
	std::vector<DataT> m_data;
};

}} // namespace mufflon::util