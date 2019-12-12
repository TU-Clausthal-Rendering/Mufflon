#include "object.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/world_container.hpp"


namespace mufflon::scene {

template < Device dev >
void Object::synchronize() {
	for(auto& lod : m_lods) {
		if(lod.has_data())
			lod.get_highest_priority_data().template synchronize<dev>();
	}
}

// Unloads all LoDs from the device
template < Device dev >
void Object::unload() {
	for(auto& lod : m_lods) {
		if(lod.has_data())
			lod.get_highest_priority_data().template unload<dev>();
	}
}

Lod& Object::get_or_fetch_original_lod(u32 level) {
	if(!has_original_lod_available(level))
		WorldContainer::instance().load_lod(*this, level);
	return get_original_lod(level);
}

template void Object::synchronize<Device::CPU>();
template void Object::synchronize<Device::CUDA>();
//template void Object::synchronize<Device::OPENGL>();
template void Object::unload<Device::CPU>();
template void Object::unload<Device::CUDA>();
//template void Object::unload<Device::OPENGL>();

} // namespace mufflon::scene
