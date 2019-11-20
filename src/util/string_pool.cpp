#include "string_pool.hpp"

namespace mufflon::util {

class StringPool::Node {
public:
	static constexpr std::size_t CHARS = NODE_PAGE_COUNT * PAGE_SIZE - (sizeof(std::size_t) + sizeof(Node*));

	Node() noexcept :
		m_pos{ 0u },
		m_next{ nullptr }
	{}
	// No move or copy - once they are placed in memory, nodes MUST stay where they are!
	Node(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(const Node&) = delete;
	Node& operator=(Node&&) = delete;
	~Node() {
		if(m_next)
			delete m_next;
	}

	char* insert(const StringView string) {
		if(m_pos + string.size() + 1u > CHARS)
			return nullptr;
		char* ptr = m_data + m_pos;
		(void)std::memcpy(ptr, string.data(), string.size());
		m_pos += string.size();
		// Important: add null termination for legacy-C usage
		*(m_data + m_pos) = '\0';
		++m_pos;
		return ptr;
	}

	Node* extent() {
		m_next = new Node();
		return m_next;
	}

private:
	std::size_t m_pos;
	Node* m_next;
	char m_data[CHARS];
};

StringPool::StringPool()
{
	static_assert(sizeof(StringPool::Node) % PAGE_SIZE == 0,
				  "The node size must be a multiple of a page size for optimal alignment");
	this->clear();
}

StringPool::StringPool(StringPool&&) = default;
StringPool& StringPool::operator=(StringPool&&) = default;
StringPool::~StringPool() = default;

StringView StringPool::insert(const StringView str) {
	if(str.size() > Node::CHARS)
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
	m_tree = std::make_unique<Node>();
	m_head = m_tree.get();
}

} // namespace mufflon::util