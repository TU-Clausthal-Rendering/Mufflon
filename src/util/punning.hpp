#pragma once

#include <cstring>
#include <type_traits>

namespace mufflon::util {

/**
 * Performs type punning for general types.
 * The types must match in size and alignment.
 * The result type must be default-constructible or, alternatively, copy-constructible,
 * in which case an optionally specified value will be used as the receiving storage.
 */
template < class U, class V >
inline U pun(const V& val, U punnedFrom = U()) {
	static_assert(sizeof(U) == sizeof(V), "Punned type must match size");
	static_assert(alignof(U) == alignof(V), "Punned type must match alignment");
	
	std::memcpy(&punnedFrom, &val, sizeof(U));
	return punnedFrom;
}

} // namespace mufflon::util