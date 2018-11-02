#include "instance.hpp"
#include "object.hpp"

namespace mufflon::scene {

Instance::Instance(Object& obj, TransMatrixType trans) :
	m_objRef(obj),
	m_transMat(std::move(trans)) {
}

const ei::Box& Instance::get_bounding_box() const noexcept {
	// TODO: transform the bounding box into oriented box
	return m_objRef.get_bounding_box();
}

} // namespace mufflon::scene