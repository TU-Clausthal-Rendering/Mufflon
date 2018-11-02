#include "scene.hpp"

namespace mufflon::scene {

bool Scene::is_accel_dirty(Device res) const noexcept {
	return m_accelDirty || m_accel_struct->is_dirty(res);
}

void Scene::clear_accel_structure() {
	// Mark as dirty only if we change something
	m_accelDirty |= m_accel_struct != nullptr;
	m_accel_struct.reset();
}

void Scene::build_accel_structure() {
	m_accelDirty = false;
	m_accel_struct->build(m_instances);
}

} // namespace mufflon::scene