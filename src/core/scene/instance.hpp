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

	const ei::Box& get_bounding_box() const noexcept;

	Object& get_object() noexcept {
		return m_objRef;
	}
	const Object& get_object() const noexcept {
		return m_objRef;
	}

	template < Device dev >
	InstanceDescriptor<dev> get_descriptor() {
		// We need to leave the objectdescriptor empty and leave it
		// up to the scene to properly fill it in, since we do not
		// yet have an array of object descriptors
		return InstanceDescriptor<dev>{
			m_transMat,
			0u
		};
	}

private:
	Object& m_objRef;
	TransMatrixType m_transMat;
};

}} // namespace mufflon::scene