#pragma once

#include "lod.hpp"
#include "util/eviction.hpp"
#include <memory>
#include "util/flag.hpp"
#include "util/string_view.hpp"
#include <optional>
#include <vector>

namespace mufflon::scene {

// Forward declaration
namespace accel_struct {
	class IAccelerationStructure;
}

struct ObjectFlags : public util::Flags<u32> {
	// NONE
};


/**
 * Representation of a scene object.
 * It is responsible for one or multiple level-of-detail as well as
 * meta-information such as animation frame.
 */
class Object {
public:
	// Available geometry types - extend if necessary
	static constexpr u32 NO_ANIMATION_FRAME = std::numeric_limits<u32>::max();

	Object(u32 objectId) : m_objectId(objectId) {}
	Object(const Object&) = delete;
	Object(Object&& obj) = default;
	Object& operator=(const Object&) = delete;
	Object& operator=(Object&&) = delete;
	~Object() = default;

	// Returns the name of the object (references the string in the object map
	// located in the world container)
	const StringView& get_name() const noexcept {
		return m_name;
	}

	// Sets the name of the object (care: since it takes a stringview, the
	// underlying string must NOT be moved/changed)
	void set_name(StringView name) noexcept {
		m_name = name;
	}

	void set_flags(ObjectFlags flags) noexcept {
		m_flags = flags;
	}
	ObjectFlags get_flags() const noexcept {
		return m_flags;
	}

	bool has_original_lod_available(u32 level) const noexcept {
		return level < m_lods.size() && m_lods[level].original.is_admitted();
	}
	bool has_reduced_lod_available(u32 level) const noexcept {
		return level < m_lods.size() && m_lods[level].reduced.is_admitted();
	}

	Lod& get_original_lod(u32 level) {
		if(!has_original_lod_available(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return *m_lods[level].original;
	}
	const Lod& get_original_lod(u32 level) const {
		if(!has_original_lod_available(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return *m_lods[level].original;
	}
	Lod& get_reduced_lod(u32 level) {
		if(!has_reduced_lod_available(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return *m_lods[level].reduced;
	}
	const Lod& get_reduced_lod(u32 level) const {
		if(!has_reduced_lod_available(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return *m_lods[level].reduced;
	}

	// This is a potentially expensive operation as it may require a disk read!
	Lod& get_or_fetch_original_lod(class WorldContainer& world, u32 level);

	// Gets the 'applicable' LoD, ie. reduced if present, else original if present
	bool has_lod(u32 level) const noexcept {
		return level < m_lods.size() && m_lods[level].has_data();
	}
	Lod& get_lod(u32 level) {
		if(!has_lod(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return m_lods[level].get_highest_priority_data();
	}
	const Lod& get_lod(u32 level) const {
		if(!has_lod(level))
			throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		return m_lods[level].get_highest_priority_data();
	}

	// Returns the number of LoD slots
	std::size_t get_lod_slot_count() const noexcept {
		return m_lods.size();
	}

	void copy_lods_from(Object& object) {
		m_lods.clear();
		for(auto& lod : object.m_lods) {
			m_lods.emplace_back();
			if(m_lods.back().original.is_admitted())
				m_lods.back().original->set_parent(this);
			if(m_lods.back().reduced.is_admitted())
				m_lods.back().reduced->set_parent(this);
		}
	}

	// Allocates the LoD slots, but keeps them evicted
	void allocate_lod_levels(u32 count) {
		if(m_lods.size() < count)
			m_lods.resize(count);
	}

	// Adds a new (or overwrites, if already existing) LoD
	Lod& add_lod(u32 level) {
		if(m_lods.size() <= level)
			m_lods.resize(level + 1u);
		return m_lods[level].original.admit(this);
	}

	Lod& add_reduced_lod(u32 level) {
		if(m_lods.size() <= level)
			m_lods.resize(level + 1u);
		if(!m_lods[level].original.is_admitted())
			return m_lods[level].reduced.admit(this);
		return m_lods[level].reduced.admit(*m_lods[level].original);
	}

	// Removes a LoD
	void remove_lod(std::size_t level) {
		if(level < m_lods.size()) {
			m_lods[level].original.evict();
			m_lods[level].reduced.evict();
		}
	}

	// Unloads the original LoD
	void remove_original_lod(std::size_t level) {
		if(level < m_lods.size())
			m_lods[level].original.evict();
	}
	void remove_reduced_lod(std::size_t level) {
		if(level < m_lods.size())
			m_lods[level].reduced.evict();
	}

	u32 get_object_id() const noexcept {
		return m_objectId;
	}

	// Synchronizes all LoDs to the device
	template < Device dev >
	void synchronize();

	// Unloads all LoDs from the device
	template < Device dev >
	void unload();

	void increase_instance_counter() noexcept {
		m_instanceCounter++;
		mAssertMsg(m_instanceCounter != 0, "Object instance counter overflow");
	}
	void decrease_instance_counter() noexcept {
		mAssertMsg(m_instanceCounter != 0, "Object instance counter underflow");
		m_instanceCounter--;
	}
	u32 get_instance_counter() const noexcept {
		return m_instanceCounter;
	}
private:
	struct LodData {
		util::Evictable<Lod> original{};
		util::Evictable<Lod> reduced{};

		bool has_data() const noexcept {
			return original.is_admitted() || reduced.is_admitted();
		}

		// Returns the highest-priority existing LoD
		Lod& get_highest_priority_data() {
			if(reduced.is_admitted())
				return *reduced;
			else if(original.is_admitted())
				return *original;
			else
				throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		}
		const Lod& get_highest_priority_data() const {
			if(reduced.is_admitted())
				return *reduced;
			else if(original.is_admitted())
				return *original;
			else
				throw std::runtime_error("Requested LOD not available. Call has_lod_available before using get_lod().");
		}
	};

	StringView m_name;
	std::vector<LodData> m_lods;
	const u32 m_objectId;

	ObjectFlags m_flags;

	u32 m_instanceCounter = 0;
};

} // namespace mufflon::scene
