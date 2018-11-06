#pragma once

#include "util/assert.hpp"
#include <memory>

namespace mufflon {

/*
 * Basic size prediction for trivial types.
 * Specialize for the specific needs of more complex types.
 * The difference to a sizeof() is that the type may have additional (runtime)
 * parameter dependent memory. I.e. if the size of an instance is sizeof(T)+x.
 */
//template<typename T, typename... Args> inline std::size_t predict_size(Args...) {
//	return sizeof(T);
//}

/*
 * Helper function for a systactical more convienient reinterpretation of memory.
 * The nice trick with overloading is that you don't need to bother with const/ref/pointer.
 */
template<typename TTarget, typename T> inline TTarget& as(T& t) {
	return reinterpret_cast<TTarget&>(t);
}
template<typename TTarget, typename T> inline const TTarget& as(const T& t) {
	return reinterpret_cast<const TTarget&>(t);
}
template<typename TTarget, typename T> inline TTarget* as(T* t) {
	return reinterpret_cast<TTarget*>(t);
}
template<typename TTarget, typename T> inline const TTarget* as(const T* t) {
	return reinterpret_cast<const TTarget*>(t);
}

} // namespace mufflon