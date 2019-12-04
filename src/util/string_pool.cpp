#include "string_pool.hpp"

namespace mufflon::util {

class StringPool::Node {
public:
	Node(std::size_t charCount) noexcept :
		m_pos{ 0u },
		m_charCount{ charCount },
		m_next{ nullptr }
	{}
	// No move or copy - once they are placed in memory, nodes MUST stay where they are!
	Node(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(const Node&) = delete;
	Node& operator=(Node&&) = delete;
	~Node() {
		if(m_next) {
			m_next->~Node();
			free(m_next);
		}
	}

	char* insert(const StringView string) {
		if(m_pos + string.size() + 1u > m_charCount)
			return nullptr;
		char* data = reinterpret_cast<char*>(this) + sizeof(Node);
		char* ptr = data + m_pos;
		(void)std::memcpy(ptr, string.data(), string.size());
		m_pos += string.size();
		// Important: add null termination for legacy-C usage
		*(data + m_pos) = '\0';
		++m_pos;
		return ptr;
	}

	Node* extent() {
		void* memory = std::malloc(sizeof(Node) + m_charCount);
		m_next = new (memory) Node{ m_charCount };
		return m_next;
	}

private:
	std::size_t m_pos;
	std::size_t m_charCount;
	Node* m_next;
};

StringPool::StringPool(const std::size_t pageCount) :
	m_tree(nullptr, &delete_head_node),
	m_poolSize{ (pageCount > 0u ? pageCount : 1u) * PAGE_SIZE }
{
	this->clear();
}

StringPool::StringPool(StringPool&&) = default;
StringPool& StringPool::operator=(StringPool&&) = default;
StringPool::~StringPool() = default;

StringView StringPool::insert(const StringView str) {
	// Only allocate once we actually have data to store
	if(!m_head)
		this->allocate_head_node();

	if(str.size() > m_poolSize - sizeof(Node))
		throw std::runtime_error("String too large to fit into a single node!");
	// TODO: this is slightly wasteful, since worst case we could do small -> large -> small,
	// leading to individual nodes even for the small strings
	char* ptr = m_head->insert(str);
	if(ptr == nullptr) {
		m_head = m_head->extent();
		ptr = m_head->insert(str);
	}
	return StringView(ptr, str.size());
}

void StringPool::clear() {
	m_tree = {};
	m_head = nullptr;
}


void StringPool::allocate_head_node() {
	// Allocate the node
	const std::size_t charCount = m_poolSize - sizeof(Node);
	void* memory = std::malloc(m_poolSize);
	Node* node = new (memory) Node{ charCount };
	m_tree.reset(node);
	m_head = node;
}

void StringPool::delete_head_node(Node* node) {
	if(node) {
		node->~Node();
		free(node);
	}
}

} // namespace mufflon::util