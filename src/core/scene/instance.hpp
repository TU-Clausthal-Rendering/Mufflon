#pragma once

#include "types.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "core/scene/descriptors.hpp"
#include <climits>

namespace mufflon { namespace scene {

class Object;

class Instance {
public:
	static constexpr u32 NO_ANIMATION_FRAME = std::numeric_limits<u32>::max();

	// TODO: identity matrix
	Instance(Object& obj, u32 index);
	Instance(const Instance&) = default;
	Instance(Instance&&) = default;
	Instance& operator=(const Instance&) = delete;
	Instance& operator=(Instance&&) = delete;
	~Instance() = default;

	u32 get_index() const noexcept { return m_index; }

	static ei::Vec3 extract_scale(const ei::Mat3x4& transformation) noexcept {
		return ei::Vec3{
			ei::len(ei::Vec<float, 3>(transformation, 0u, 0u)),
			ei::len(ei::Vec<float, 3>(transformation, 0u, 1u)),
			ei::len(ei::Vec<float, 3>(transformation, 0u, 2u))
		};
	}

	ei::Box get_bounding_box(u32 lod, const ei::Mat3x4& transformation) const noexcept;

	Object& get_object() noexcept {
		return *m_objRef;
	}
	const Object& get_object() const noexcept {
		return *m_objRef;
	}
	void set_object(Object& object) noexcept;


private:
	Object* m_objRef;
	u32 m_index;
};

}} // namespace mufflon::scene
