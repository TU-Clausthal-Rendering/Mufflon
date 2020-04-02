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

// Checks if two sorted ranges share elements or not
template < class I1, class I2 >
constexpr bool share_elements_sorted(const I1 beginA, const I1 endA, const I2 beginB, const I2 endB) noexcept {
	auto currA = beginA;
	auto currB = beginB;
	while(currA != endA && currB != endB) {
		if(*currA == *currB)
			return true;
		if(*currA < *currB)
			++currA;
		else
			++currB;
	}
	return false;
}

}} // namespace mufflon::util