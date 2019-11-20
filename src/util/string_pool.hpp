#pragma once

#include "string_view.hpp"
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>

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

} // namespace mufflon::util