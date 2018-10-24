#pragma once

#include "object.hpp"
#include "ei/vector.hpp"

namespace mufflon::scene {

class Instance {
public:
	using TransMatrixType = ei::Matrix<Real, 4, 3>;

	Instance(Object& obj, TransMatrixType trans = {}) :
		m_objRef(obj),
		m_transMat(std::move(trans))
	{}

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

private:
	Object& m_objRef;
	TransMatrixType m_transMat;
};

} // namespace mufflon::scene