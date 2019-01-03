#include "generic_resource.hpp"
#include "synchronize.hpp"

namespace mufflon {

void GenericResource::resize(std::size_t size) {
	// Release all resources if they have the wrong size.
	if(m_size != size) {
		m_mem.template get<unique_device_ptr<Device::CPU, char[]>>() = nullptr;
		m_mem.template get<unique_device_ptr<Device::CUDA, char[]>>() = nullptr;
	}
	// Set the size for future allocations
	m_size = size;
}

template < Device dstDev >
void GenericResource::synchronize() {
	if(m_mem.template get<unique_device_ptr<dstDev, char[]>>() == nullptr)
		m_mem.template get<unique_device_ptr<dstDev, char[]>>() = make_udevptr_array<dstDev, char>(m_size);
	if(m_dirty.needs_sync(dstDev) && m_size != 0u) {	// Otherwise we would do a useless copy inside the same memory
		const char* srcDev = nullptr;
		if(m_dirty.has_changes(Device::CPU))
			srcDev = m_mem.template get<unique_device_ptr<Device::CPU, char[]>>().get();
		if(m_dirty.has_changes(Device::CUDA))
			srcDev = m_mem.template get<unique_device_ptr<Device::CUDA, char[]>>().get();
		mAssertMsg(srcDev != nullptr, "Device not supported or DirtyFlags inconsistent.");
		copy(m_mem.template get<unique_device_ptr<dstDev, char[]>>().get(),
			 srcDev, m_size);
		m_dirty.mark_synced(dstDev);
	}
}

// Explicit instanciations
template void GenericResource::synchronize<Device::CPU>();
template void GenericResource::synchronize<Device::CUDA>();

} // namespace mufflon
