#pragma once

#include "util/assert.hpp"
#include "core/export/api.h"
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
	CUDA_FUNCTION void set(T flag) noexcept { mask = mask | flag; }
	// Remove a flag, may set or remove multiple flags at once
	CUDA_FUNCTION void clear(T flag) noexcept { mask = mask & ~flag; }
	// Remove all flags (initial state)
	CUDA_FUNCTION void clear_all() noexcept { mask = 0; }
	// Remove or set a flag based on the current state, may set or remove multiple flags at once
	CUDA_FUNCTION void toggle(T flag) noexcept { mask = mask ^ flag; }
	// Check if a specific flag is set
	CUDA_FUNCTION bool is_set(T flag) const noexcept {
		mAssertMsg(is_power_of_two(flag),
				   "Only a single flag (bit) should be checked.");
		return (mask & flag) != 0;
	}
	// Check if NO flag is set
	CUDA_FUNCTION bool is_empty() const noexcept { return mask == 0; }
	// Check if exactly one flag is set
	CUDA_FUNCTION bool is_unique() const noexcept { return is_power_of_two(mask); }

	CUDA_FUNCTION operator T () const noexcept { return mask; }
};

/**
 * A kind of bitset which allows for change tracking of attributes.
 */
template < class E >
class DirtyFlags {
public:
	using Enum = E;
	using EnumType = std::underlying_type_t<Enum>;

	DirtyFlags() = default;
	DirtyFlags(const DirtyFlags&) = default;
	DirtyFlags(DirtyFlags&&) = default;
	DirtyFlags& operator=(const DirtyFlags&) = default;
	DirtyFlags& operator=(DirtyFlags&&) = default;
	~DirtyFlags() = default;

	// Marks a value as changed - all other values will be marked as subject-to-change
	void mark_changed(Enum from) noexcept {
		EnumType value = get_value(from);
		// Check if the changed value has been synchronized before -> no issues
		if(!(m_needsSyncing & value)) {
			// This value now aggregates all changes; future syncs should
			// pull from this one
			m_hasChanges = value;
		} else {
			// Not good; the value changed before synchronizing -> keep both changes
			m_hasChanges |= value;
		}

		m_needsSyncing = ~value;
	}

	/**
	 * Marks a value as synchronized. This should be done for all values if
	 * one gets marked as changed.
	 */
	void mark_synced(Enum check) noexcept {
		EnumType value = get_value(check);
		m_needsSyncing &= ~value;
	}

	// Checks whether a value needs to synchronize.
	bool needs_sync(Enum check) const noexcept {
		EnumType value = get_value(check);
		return m_needsSyncing & value;
	}

	constexpr bool has_changes() const noexcept {
		return m_hasChanges != 0u;
	}

	// Checks if the given value has changes
	bool has_changes(Enum check) const noexcept {
		EnumType value = get_value(check);
		return m_hasChanges & value;
	}

	// Checks if two values have been marked as changed at the same time.
	bool has_competing_changes() const noexcept {
		return m_needsSyncing & m_hasChanges;
	}

	// Marks a value as absent on a device
	void unload(Enum check) noexcept {
		if(this->is_present(check)) {
			EnumType value = get_value(check);
			// Mark as absent as well as in need of sync if it was ever present
			m_isPresent &= ~value;
			m_needsSyncing |= value;
		}
	}

	// Checks if a value is present on a device at all
	bool is_present(Enum check) const noexcept {
		EnumType value = get_value(check);
		return m_isPresent & value;
	}

private:
	static EnumType get_value(Enum e) noexcept {
		EnumType value = static_cast<EnumType>(e);
		// Make sure that only one bit is set, i.e. the value is a power of two
		mAssertMsg(is_power_of_two(value),
				   "Enums for bitfields must not share bits between values");
		return value;
	}

	EnumType m_needsSyncing = static_cast<EnumType>(0u);
	EnumType m_hasChanges = static_cast<EnumType>(0u);
	EnumType m_isPresent = static_cast<EnumType>(0u);
};

}} // namespace mufflon::util