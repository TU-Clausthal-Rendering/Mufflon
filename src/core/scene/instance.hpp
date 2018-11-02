#pragma once

#include "object.hpp"
#include "ei/vector.hpp"
#include "ei/3dtypes.hpp"

namespace mufflon::scene {

class Instance {
public:
	using TransMatrixType = ei::Matrix<Real, 4, 3>;

	Instance(Object& obj, TransMatrixType trans = {});
	Instance(const Instance&) = default;
	Instance(Instance&&) = default;
	Instance& operator=(const Instance&) = delete;
	Instance& operator=(Instance&&) = delete;
	~Instance() = default;

	void set_transformation_matrix(TransMatrixType mat) {
		m_transMat = std::move(mat);
	}

	const TransMatrixType& get_transformation_matrix() const noexcept {
		return m_transMat;
	}

	const ei::Box& get_bounding_box() const noexcept {
		// TODO: transform the bounding box into oriented box
		return m_objRef.get_bounding_box();
	}

private:
	Object& m_objRef;
	TransMatrixType m_transMat;
};

} // namespace mufflon::scene