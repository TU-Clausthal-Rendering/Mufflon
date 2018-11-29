#include "scene.hpp"
#include "core/scene/accel_structs/accel_struct.hpp"
#include "core/scene/materials/medium.hpp"

namespace mufflon::scene {

void Scene::load_media(const std::vector<materials::Medium>& media) {
	m_media.resize(sizeof(materials::Medium) * media.size());
	materials::Medium* dst = m_media.get<Device::CPU, materials::Medium>();
	memcpy(dst, media.data(), m_media.size());
}

bool Scene::is_accel_dirty(Device res) const noexcept {
	return m_accelDirty || m_accelStruct->is_dirty(res);
}

void Scene::clear_accel_structure() {
	// Mark as dirty only if we change something
	m_accelDirty |= m_accelStruct != nullptr;
	m_accelStruct.reset();
}

void Scene::build_accel_structure() {
	m_accelDirty = false;
	m_accelStruct->build(m_instances);
}

} // namespace mufflon::scene