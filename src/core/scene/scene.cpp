#include "scene.hpp"
#include "core/scene/accel_structs/accel_struct.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/materials/material.hpp"

namespace mufflon::scene {

void Scene::load_media(const std::vector<materials::Medium>& media) {
	m_media.resize(sizeof(materials::Medium) * media.size());
	materials::Medium* dst = m_media.acquire<Device::CPU, materials::Medium>();
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


template< Device dev >
void Scene::load_materials() {
	// 1. Pass get the sizes for the index -> material offset table
	std::vector<int> offsets;
	std::size_t offset = sizeof(int) * m_materialsRef.size(); // Store in one block -> table size is offset of first material
	for(const auto& mat : m_materialsRef) {
		mAssert(offset <= std::numeric_limits<i32>::max());
		offsets.push_back(i32(offset));
		//offset += materials::get_handle_pack_size();
		offset += mat->get_handle_pack_size(dev);
	}
	// Allocate the memory
	m_materials.resize(offset);
	char* mem = m_materials.acquire<dev, char>();
	copy<dev,Device::CPU>(mem, offsets.data(), sizeof(int) * m_materialsRef.size());
	// 2. Pass get all the material descriptors
	char buffer[materials::MAX_MATERIAL_PARAMETER_SIZE];
	int i = 0;
	for(const auto& mat : m_materialsRef) {
		mAssert(mat->get_handle_pack_size(dev) <= materials::MAX_MATERIAL_PARAMETER_SIZE);
		mat->get_handle_pack(dev, as<materials::HandlePack>(buffer));
		copy<dev,Device::CPU>(mem + offsets[i], buffer, mat->get_handle_pack_size(dev));
		++i;
	}
}

template void Scene::load_materials<Device::CPU>();
template void Scene::load_materials<Device::CUDA>();

} // namespace mufflon::scene