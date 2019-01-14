#include "instance.hpp"
#include "object.hpp"

namespace mufflon::scene {

Instance::Instance(Object& obj, TransMatrixType trans) :
	m_objRef(obj) {
	this->set_transformation_matrix(std::move(trans));
}

ei::Box Instance::get_bounding_box() const noexcept {
	return transform(m_objRef.get_bounding_box(), m_transMat);
}

} // namespace mufflon::scene