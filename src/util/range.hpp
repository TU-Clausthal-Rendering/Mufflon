#pragma once

namespace mufflon { namespace util {

/**
 * Represents a range of iterators.
 * May be used in a for-each loop. Ideal if an object only provides iterators, not begin/end,
 * or if one does not want to iterate the full range.
 */
template < class Iter >
class Range {
public:
	using Iterator = Iter;

	Range(Iterator begin, Iterator end) :
		m_begin(std::move(begin)),
		m_end(std::move(end)) {}

	Iterator& begin() {
		return m_begin;
	}
	const Iterator& cbegin() {
		return m_begin;
	}

	Iterator& end() {
		return m_end;
	}
	const Iterator& cend() {
		return m_end;
	}

private:
	Iterator m_begin;
	Iterator m_end;
};

}} // namespace mufflon::util