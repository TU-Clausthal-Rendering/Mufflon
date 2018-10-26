#include "object.hpp"
#include "accel_struct.hpp"

namespace mufflon::scene {


bool Object::is_accel_dirty(Device res) const noexcept {
	return m_accelDirty || m_accel_struct->is_dirty(res);
}

void Object::build_accel_structure() {
	// We no longer need this indication - the structure itself will tell us
	// if and where we are dirty
	m_accelDirty = false;

	m_accel_struct->build();
}

} // namespace mufflon::scene