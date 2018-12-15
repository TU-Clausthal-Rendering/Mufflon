#include "object.hpp"
#include "core/scene/descriptors.hpp"

namespace mufflon::scene {

Object::Object() {
}

Object::Object(Object&& obj) :
	m_name(obj.m_name),
	m_geometryData(std::move(obj.m_geometryData)),
	m_accelStruct(),
	m_animationFrame(obj.m_animationFrame),
	m_lodLevel(obj.m_lodLevel),
	m_flags(obj.m_flags)
{

}

Object::~Object() {

}

void Object::clear_accel_structure() {
	m_accelStruct[get_device_index<Device::CPU>()].type = accel_struct::AccelType::NONE;
	m_accelStruct[get_device_index<Device::CUDA>()].type = accel_struct::AccelType::NONE;
	// TODO memory
}

} // namespace mufflon::scene
