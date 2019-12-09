#pragma once

#include "string_view.hpp"
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <map>

namespace mufflon::util {

class StringPool {
public:
	static constexpr std::size_t PAGE_SIZE = 0x1000u; // Most common OS page size
	// Default is 128KB per pool - should be enough for simple scenes
	static constexpr std::size_t DEFAULT_PAGE_COUNT = 32u;

	StringPool(const std::size_t pageCount = DEFAULT_PAGE_COUNT);
	StringPool(const StringPool&) = delete;
	StringPool(StringPool&&);
	StringPool& operator=(const StringPool&) = delete;
	StringPool& operator=(StringPool&&);
	~StringPool();

	// Careful - this method is NOT thread-safe! Should the need for this arise, make sure
	// to add a synchronization primitive to the insertion
	StringView insert(const StringView str);
	void clear();
	bool empty() const noexcept { return m_head == nullptr; }

private:
	class Node;

	void allocate_head_node();
	static void delete_head_node(Node*);

	std::unique_ptr<Node, void(*)(Node*)> m_tree;
	Node* m_head;
	std::size_t m_poolSize;
};

// This string pool makes sense when you have a few unique names that you want to share.
// Since it has the ability to shrink, we make it global
class UniqueStringPool {
public:
	// WARNING: this is NOT thread-safe!
	StringView insert(StringView str) {
		if(const auto iter = m_strings.find(str); iter != m_strings.end()) {
			++iter->second;
			return iter->first;
		} else {
			return m_strings.emplace(std::string(str), 1u).first->first;
		}
	}
	// WARNING: this is NOT thread-safe!
	void remove(StringView str) {
		if(const auto iter = m_strings.find(str); iter != m_strings.end())
			if(--iter->second == 0u)
				m_strings.erase(iter);
	}

	static UniqueStringPool& instance() {
		static UniqueStringPool inst{};
		return inst;
	}

private:
	UniqueStringPool() = default;
	UniqueStringPool(const UniqueStringPool&) = delete;
	UniqueStringPool(UniqueStringPool&&) = default;
	UniqueStringPool& operator=(const UniqueStringPool&) = delete;
	UniqueStringPool& operator=(UniqueStringPool&&) = default;
	~UniqueStringPool() = default;

	// TODO: Find a way to query an unordered map of strings with string_views,
	// while still not invalidating references on insert
	std::map<std::string, std::size_t, std::less<>> m_strings;
};

} // namespace mufflon::util
