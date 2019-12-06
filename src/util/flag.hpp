#pragma once

#include "util/assert.hpp"
#include "core/export/core_api.h"
#include <climits>
#include <type_traits>

namespace mufflon { namespace util {

// Check if an integer is a power of two.
template < class T >
static constexpr bool is_power_of_two(T value) noexcept {
	return value && !(value & (value - 1));
}

/**
 * Basis type to implement custom flags with useful operations.
 * Enums (class and legacy) both defy the usecase as flags because they require
 * repeated castings.
 * T: an integer type, requires logic operations
 *
 * Usage: inherit a struct with static constexpr members of type T for the flags.
 */
template < class T >
struct Flags {
	using BasicType = T;
	T mask = 0;

	// Set a flag, may set or remove multiple flags at once
	inline CUDA_FUNCTION void set(T flag) noexcept { mask = mask | flag; }
	// Remove a flag, may set or remove multiple flags at once
	inline CUDA_FUNCTION void clear(T flag) noexcept { mask = mask & ~flag; }
	// Remove all flags (initial state)
	inline CUDA_FUNCTION void clear_all() noexcept { mask = 0; }
	// Remove or set a flag based on the current state, may set or remove multiple flags at once
	inline CUDA_FUNCTION void toggle(T flag) noexcept { mask = mask ^ flag; }
	// Check if a specific flag is set
	inline CUDA_FUNCTION bool is_set(T flag) const noexcept {
		mAssertMsg(is_power_of_two(flag),
				   "Only a single flag (bit) should be checked.");
		return (mask & flag) != 0;
	}
	// Check if one of a number of flags is set
	inline CUDA_FUNCTION bool is_any_set(T mask) const noexcept {
		return (this->mask & mask) != 0;
	}
	// Check if NO flag is set
	inline CUDA_FUNCTION bool is_empty() const noexcept { return mask == 0; }
	// Check if exactly one flag is set
	inline CUDA_FUNCTION bool is_unique() const noexcept { return is_power_of_two(mask); }

	constexpr inline CUDA_FUNCTION operator T () const noexcept { return mask; }
};

}} // namespace mufflon::util