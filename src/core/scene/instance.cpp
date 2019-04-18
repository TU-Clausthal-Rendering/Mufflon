#include "instance.hpp"
#include "object.hpp"

namespace mufflon::scene {

Instance::Instance(std::string name, Object& obj, ei::Mat3x4 trans) :
	m_name(move(name)),
	m_objRef(&obj)
{
	this->set_transformation_matrix(std::move(trans));
	m_objRef->increase_instance_counter();
}

ei::Box Instance::get_bounding_box(u32 lod) const noexcept {
	return transform(m_objRef->get_lod(lod).get_bounding_box(), m_transMat * ei::scaling(ei::Vec4{m_scale, 1.0f}));
}

void Instance::set_object(Object& object) noexcept
{
	m_objRef->decrease_instance_counter();
	m_objRef = &object;
	m_objRef->increase_instance_counter();
}
} // namespace mufflon::scene
