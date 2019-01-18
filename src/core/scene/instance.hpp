#pragma once

#include "types.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"
#include "core/scene/descriptors.hpp"

namespace mufflon { namespace scene {

class Object;

class Instance {
public:
	using TransMatrixType = ei::Matrix<Real, 3, 4>;

	// TODO: identity matrix
	Instance(std::string name, Object& obj, TransMatrixType trans = {});
	Instance(const Instance&) = default;
	Instance(Instance&&) = default;
	Instance& operator=(const Instance&) = delete;
	Instance& operator=(Instance&&) = delete;
	~Instance() = default;

	void set_transformation_matrix(TransMatrixType mat) {
		mAssertMsg(ei::approx(ei::len(mat(0u).subrow<0u, 3u>()), ei::len(mat(1u).subrow<0u, 3u>()))
				   && ei::approx(ei::len(mat(0u).subrow<0u, 3u>()), ei::len(mat(2u).subrow<0u, 3u>())),
				   "Instance transformations must have uniform scale");
		m_transMat = std::move(mat);
		m_scale = ei::len(m_transMat(0u).subrow<0u, 3u>());
		m_transMat(0u).subrow<0u, 3u>() *= 1.f / m_scale;
		m_transMat(1u).subrow<0u, 3u>() *= 1.f / m_scale;
		m_transMat(2u).subrow<0u, 3u>() *= 1.f / m_scale;
	}

	const TransMatrixType& get_transformation_matrix() const noexcept {
		return m_transMat;
	}

	float get_scale() const noexcept {
		return m_scale;
	}

	ei::Box get_bounding_box() const noexcept;

	Object& get_object() noexcept {
		return m_objRef;
	}
	const Object& get_object() const noexcept {
		return m_objRef;
	}

private:
	std::string m_name;
	Object& m_objRef;
	TransMatrixType m_transMat;
	float m_scale;
};

}} // namespace mufflon::scene