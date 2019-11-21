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
	static constexpr std::size_t NODE_PAGE_COUNT = 16u; // 128KB (minus bookkeeping)
	static constexpr std::size_t PAGE_SIZE = 0x1000u; // Most common OS page size

	static_assert(NODE_PAGE_COUNT > 0u, "We need at least one page");

	StringPool();
	StringPool(const StringPool&) = delete;
	StringPool(StringPool&&);
	StringPool& operator=(const StringPool&) = delete;
	StringPool& operator=(StringPool&&);
	~StringPool();

	// Careful - this method is NOT thread-safe! Should the need for this arise, make sure
	// to add a synchronization primitive to the insertion
	StringView insert(const StringView str);
	void clear();

private:
	class Node;

	std::unique_ptr<Node> m_tree;
	Node* m_head;
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