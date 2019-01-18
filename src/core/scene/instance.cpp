#include "instance.hpp"
#include "object.hpp"

namespace mufflon::scene {

Instance::Instance(std::string name, Object& obj, const ei::Mat3x4& trans) :
	m_name(move(name)),
	m_objRef(obj) {
	this->set_transformation_matrix(trans);
}

ei::Box Instance::get_bounding_box() const noexcept {
	return transform(m_objRef.get_bounding_box(), m_transMat);
}

} // namespace mufflon::scene