#include "instance.hpp"

namespace mufflon::scene {

Instance::Instance(Object& obj, TransMatrixType trans) :
	m_objRef(obj),
	m_transMat(std::move(trans)) {
}

} // namespace mufflon::scene