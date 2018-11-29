#include "generic_resource.hpp"
#include "synchronize.hpp"

namespace mufflon {

void GenericResource::resize(std::size_t size) {
	// Release all resources if they have the wrong size.
	if(m_size == size) {
		m_mem.template get<unique_device_ptr<Device::CPU, char>>() = nullptr;
		m_mem.template get<unique_device_ptr<Device::CUDA, char>>() = nullptr;
	}
	// Set the size for future allocations
	m_size = size;
}

template < Device dstDev, Device srcDev >
void GenericResource::synchronize() {
	if(dstDev != srcDev) {	// Otherwise we would do a useless copy inside the same memory
		copy<dstDev, srcDev>(
			m_mem.template get<unique_device_ptr<dstDev, char>>().get(),
			m_mem.template get<unique_device_ptr<srcDev, char>>().get(),
			m_size
		);
	}
}

// Explicit instanciations
template void GenericResource::synchronize<Device::CPU, Device::CPU>();
template void GenericResource::synchronize<Device::CPU, Device::CUDA>();
template void GenericResource::synchronize<Device::CUDA, Device::CPU>();
template void GenericResource::synchronize<Device::CUDA, Device::CUDA>();

} // namespace mufflon
