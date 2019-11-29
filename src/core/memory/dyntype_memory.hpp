#pragma once

#include "core/export/api.h"

namespace mufflon { // There is no memory namespace on purpose

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
template<typename TTarget, typename T>
inline CUDA_FUNCTION TTarget& as(T& t) {
	return reinterpret_cast<TTarget&>(t);
}
template<typename TTarget, typename T>
inline CUDA_FUNCTION const TTarget& as(const T& t) {
	return reinterpret_cast<const TTarget&>(t);
}
template<typename TTarget, typename T>
inline CUDA_FUNCTION TTarget* as(T* t) {
	return reinterpret_cast<TTarget*>(t);
}
template<typename TTarget, typename T>
inline CUDA_FUNCTION const TTarget* as(const T* t) {
	return reinterpret_cast<const TTarget*>(t);
}

// Aligns up a given value (only for alignments of power of two)
template < std::size_t ALIGNMENT, class T >
inline CUDA_FUNCTION constexpr T round_to_align(T s) {
	static_assert(!(ALIGNMENT & (ALIGNMENT - 1)), "Alignment only works for powers of 2");
	return (s + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
}

} // namespace mufflon