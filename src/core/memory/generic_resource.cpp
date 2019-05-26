#include "generic_resource.hpp"
#include "synchronize.hpp"

namespace mufflon {

void GenericResource::resize(std::size_t size) {
	// Release all resources if they have the wrong size.
	if(m_size != size) {
		unload<Device::CPU>();
		unload<Device::CUDA>();
		unload<Device::OPENGL>();
	}
	// Set the size for future allocations
	m_size = size;
}

template < Device dstDev >
void GenericResource::synchronize() {
	auto& dstMem = m_mem.template get<SyncedDevPtr<dstDev>>();
	if(dstMem.ptr == nullptr && m_size != 0u) {
		dstMem.ptr = make_udevptr_array<dstDev, char>(m_size);
		dstMem.synced = false;
	}

	if(!dstMem.synced) {
		bool synced = false;
		m_mem.for_each([&](auto& data) {
			using ChangedBuffer = std::decay_t<decltype(data)>;
			if(!synced && ChangedBuffer::DEVICE != dstDev && data.synced && data.ptr != nullptr) {
				copy<char>(dstMem.ptr.get(), data.ptr.get(), m_size);
				synced = true;
			}
		});
		dstMem.synced = true;
	}
}

void GenericResource::mark_changed(Device dev) noexcept {
	if(dev != Device::CPU) unload<Device::CPU>();
	if(dev != Device::CUDA) unload<Device::CUDA>();
	if(dev != Device::OPENGL) unload<Device::OPENGL>();
}

// Explicit instanciations
template void GenericResource::synchronize<Device::CPU>();
template void GenericResource::synchronize<Device::CUDA>();
template void GenericResource::synchronize<Device::OPENGL>();

} // namespace mufflon
