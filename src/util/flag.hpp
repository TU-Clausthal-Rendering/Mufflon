#pragma once

#include "util/assert.hpp"
#include <climits>
#include <type_traits>

namespace mufflon::util {

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

private:
	static EnumType get_value(Enum e) noexcept {
		EnumType value = static_cast<EnumType>(e);
		// Make sure that only one bit is set, i.e. the value is a power of two
		mAssertMsg(is_power_of_two(value),
				   "Enums for bitfields must not share bits between values");
		return value;
	}

	static constexpr bool is_power_of_two(EnumType value) noexcept {
		return value && !(value & (value - 1));
	}

	EnumType m_needsSyncing;
	EnumType m_hasChanges;
};

} // namespace mufflon::util