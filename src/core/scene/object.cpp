#include "object.hpp"
#include "accell_struct.hpp"

namespace mufflon::scene {

bool Object::is_data_dirty(Residency res) const noexcept {
	switch(res) {
		case Residency::CPU: return m_cpuData.m_isDirty;
		case Residency::CUDA: return m_cudaData.m_isDirty;
		case Residency::OPENGL: return m_openGlData.m_isDirty;
	}
	return false;
}


bool Object::is_accell_dirty(Residency res) const noexcept {
	return m_accellDirty || m_accell_struct->is_dirty(res);
}

void Object::build_accell_structure() {
	// We no longer need this indication - the structure itself will tell us
	// if and where we are dirty
	m_accellDirty = false;

	m_accell_struct->build();
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