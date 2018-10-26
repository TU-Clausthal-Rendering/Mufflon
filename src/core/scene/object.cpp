#include "object.hpp"
#include "accel_struct.hpp"

namespace mufflon::scene {

Object::~Object() = default;

bool Object::is_data_dirty(Device res) const noexcept {
	switch(res) {
		case Device::CPU: return m_cpuData.isDirty;
		case Device::CUDA: return m_cudaData.isDirty;
		case Device::OPENGL: return m_openGlData.isDirty;
	}
	return false;
}


bool Object::is_accel_dirty(Device res) const noexcept {
	return m_accelDirty || m_accel_struct->is_dirty(res);
}

void Object::clear_accel_structutre() {
	// Mark as dirty only if we change something
	m_accelDirty |= m_accel_struct != nullptr;
	m_accel_struct.reset();
}

void Object::build_accel_structure() {
	// We no longer need this indication - the structure itself will tell us
	// if and where we are dirty
	m_accelDirty = false;

	m_accel_struct->build();
}

void Object::make_resident(Device res) {
	// TODO
	throw std::runtime_error("make_resident is not implemented yet!");
}

void Object::unload_resident(Device res) {
	// TODO
	throw std::runtime_error("unload_resident is not implemented yet!");
}

} // namespace mufflon::scene