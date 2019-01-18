#pragma once

#include "lod.hpp"
#include <memory>
#include <string_view>
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
	const std::string_view& get_name() const noexcept {
		return m_name;
	}

	// Sets the name of the object (care: since it takes a stringview, the
	// underlying string must NOT be moved/changed)
	void set_name(std::string_view name) noexcept {
		m_name = name;
	}

	void set_flags(ObjectFlags flags) noexcept {
		m_flags = flags;
	}

	// Returns the object's animation frame.
	u32 get_animation_frame() const noexcept {
		return m_animationFrame;
	}
	// Sets the object's animation frame.
	void set_animation_frame(u32 frame) noexcept {
		m_animationFrame = frame;
	}

	bool has_lod_available(u32 level) const noexcept {
		return level < m_lods.size() && m_lods[level] != nullptr;
	}

	Lod& get_lod(u32 level) noexcept {
		return *m_lods[level];
	}
	const Lod& get_lod(u32 level) const noexcept {
		return *m_lods[level];
	}

	// Returns the number of LoD slots
	std::size_t get_lod_slot_count() const noexcept {
		return m_lods.size();
	}

	// Adds a new (or overwrites, if already existing) LoD
	Lod& add_lod(u32 level) {
		if(m_lods.size() <= level)
			m_lods.resize(level + 1u);
		m_lods[level] = std::make_unique<Lod>();
		return *m_lods[level];
	}

	// Removes a LoD
	void remove_lod(std::size_t level) {
		if(level < m_lods.size())
			m_lods[level].reset();
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

private:
	std::string_view m_name;
	std::vector<std::unique_ptr<Lod>> m_lods;
	const u32 m_objectId;

	u32 m_animationFrame = NO_ANIMATION_FRAME; // Current frame of a possible animation
	ObjectFlags m_flags;

	// TODO: how to handle the LoDs?
};

} // namespace mufflon::scene
