#include "object.hpp"
#include "accel_struct.hpp"

namespace mufflon::scene {

bool Object::is_data_dirty(Residency res) const noexcept {
	switch(res) {
		case Residency::CPU: return m_cpuData.isDirty;
		case Residency::CUDA: return m_cudaData.isDirty;
		case Residency::OPENGL: return m_openGlData.isDirty;
	}
	return false;
}


bool Object::is_accel_dirty(Residency res) const noexcept {
	return m_accelDirty || m_accel_struct->is_dirty(res);
}

void Object::build_accel_structure() {
	// We no longer need this indication - the structure itself will tell us
	// if and where we are dirty
	m_accelDirty = false;

	m_accel_struct->build();
}

void Object::make_resident(Residency res) {
	// TODO
	throw std::runtime_error("make_resident is not implemented yet!");
}

void Object::unload_resident(Residency res) {
	// TODO
	throw std::runtime_error("unload_resident is not implemented yet!");
}

} // namespace mufflon::scene