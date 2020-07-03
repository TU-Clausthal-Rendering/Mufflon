#include "instance.hpp"
#include "object.hpp"

namespace mufflon::scene {

Instance::Instance(Object& obj, u32 index) :
	m_objRef(&obj),
	m_index{ index }
{
	m_objRef->increase_instance_counter();
}

ei::Box Instance::get_bounding_box(u32 lod, const ei::Mat3x4& transformation) const noexcept {
	Lod& lodRef = m_objRef->get_lod(lod);
	return transform(lodRef.get_bounding_box(), transformation);
}

void Instance::set_object(Object& object) noexcept
{
	m_objRef->decrease_instance_counter();
	m_objRef = &object;
	m_objRef->increase_instance_counter();
}
} // namespace mufflon::scene
