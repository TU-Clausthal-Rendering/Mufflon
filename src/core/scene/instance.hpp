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
	Instance(std::string name, Object& obj, ei::Mat3x4 trans = {
				1.f, 0.f, 0.f, 0.f,
				0.f, 1.f, 0.f, 0.f,
				0.f, 0.f, 1.f, 0.f 
			 });
	Instance(const Instance&) = default;
	Instance(Instance&&) = default;
	Instance& operator=(const Instance&) = delete;
	Instance& operator=(Instance&&) = delete;
	~Instance() = default;

	StringView get_name() const noexcept {
		return m_name;
	}

	void set_transformation_matrix(const ei::Mat3x4& mat) {
		m_transMat = mat;
		m_scale.x = ei::len(ei::Vec<float, 3>(m_transMat, 0u, 0u));
		m_scale.y = ei::len(ei::Vec<float, 3>(m_transMat, 0u, 1u));
		m_scale.z = ei::len(ei::Vec<float, 3>(m_transMat, 0u, 2u));
		ei::Mat4x4 invRS = invert(ei::Mat4x4{mat});
		m_invTransMat = ei::Mat3x4 { invRS };
	}

	const ei::Mat3x4& get_transformation_matrix() const noexcept {
		return m_transMat;
	}

	const ei::Mat3x4& get_inverse_transformation_matrix() const noexcept {
		return m_invTransMat;
	}

	ei::Vec3 get_scale() const noexcept {
		return m_scale;
	}

	ei::Box get_bounding_box(u32 lod) const noexcept;

	Object& get_object() noexcept {
		return *m_objRef;
	}
	const Object& get_object() const noexcept {
		return *m_objRef;
	}
	void set_object(Object& object) noexcept;

	u32 get_animation_frame() const noexcept {
		return m_animationFrame;
	}
	void set_animation_frame(const u32 animationFrame) noexcept {
		m_animationFrame = animationFrame;
	}

private:
	std::string m_name;
	Object* m_objRef;
	ei::Mat3x4 m_transMat;
	ei::Mat3x4 m_invTransMat;
	ei::Vec3 m_scale;
	u32 m_animationFrame;
};

}} // namespace mufflon::scene
