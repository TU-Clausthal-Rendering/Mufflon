#include "attribute.hpp"

namespace mufflon::scene {

// Synchronization specialization from CPU to CUDA (owning)
template <>
void AttributePool<Device::CPU, true>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool) {
	// TODO
	throw std::runtime_error("Synchronize between CPU and CUDA not yet implemented");
}

// Synchronization specialization from CPU to CUDA (non-owning)
template <>
void AttributePool<Device::CPU, false>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool) {
	// TODO
	throw std::runtime_error("Synchronize between CPU and CUDA not yet implemented");
}

// Synchronization specialization from CUDA to CPU (owning)
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, true>(AttributePool<Device::CPU, true>& pool) {
	// TODO
	throw std::runtime_error("Synchronize between CPU and CUDA not yet implemented");
}

// Synchronization specialization from CUDA to CPU (non-owning)
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, false>(AttributePool<Device::CPU, false>& pool) {
	// TODO
	throw std::runtime_error("Synchronize between CPU and CUDA not yet implemented");
}

} // namespace mufflon::scene