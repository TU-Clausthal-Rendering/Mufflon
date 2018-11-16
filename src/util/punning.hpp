#pragma once

#include <cstring>
#include <type_traits>

namespace mufflon { namespace util {

// Helper functions to print the sizes/alignment on mismatch in a static_assert
template < class U, class V, std::size_t US = sizeof(U), std::size_t VS = sizeof(V) >
inline constexpr void check_size() {
	static_assert(US == VS, "Object size mismatch");
}
template < class U, class V, std::size_t US = alignof(U), std::size_t VS = alignof(V) >
inline constexpr void check_alignment() {
	static_assert(US == VS, "Object alignment mismatch");
}

/**
 * Performs type punning for general types.
 * The types must match in size and alignment.
 * The result type must be default-constructible or, alternatively, copy-constructible,
 * in which case an optionally specified value will be used as the receiving storage.
 */
template < class U, class V >
inline U pun(const V& val, U punnedFrom = U()) {
	check_size<U, V>();
	check_alignment<U, V>();

	std::memcpy(&punnedFrom, &val, sizeof(U));
	return punnedFrom;
}

}} // namespace mufflon::util