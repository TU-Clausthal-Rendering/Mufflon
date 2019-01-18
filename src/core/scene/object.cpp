#include "object.hpp"
#include "profiler/cpu_profiler.hpp"
#include "core/scene/scenario.hpp"
#include "core/scene/materials/material.hpp"


namespace mufflon::scene {

template < Device dev >
void Object::synchronize() {
	for(auto& lod : m_lods) {
		if(lod != nullptr)
			lod->template synchronize<dev>();
	}
}

// Unloads all LoDs from the device
template < Device dev >
void Object::unload() {
	for(auto& lod : m_lods) {
		if(lod != nullptr)
			lod->template unload<dev>();
	}
}

template void Object::synchronize<Device::CPU>();
template void Object::synchronize<Device::CUDA>();
//template void Object::synchronize<Device::OPENGL>();
template void Object::unload<Device::CPU>();
template void Object::unload<Device::CUDA>();
//template void Object::unload<Device::OPENGL>();

} // namespace mufflon::scene
