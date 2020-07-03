#include "generic_resource.hpp"
#include "core/concepts.hpp"

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

template < Device dev >
ArrayDevHandle_t<dev, char> GenericResource::acquire(bool sync) {
	if(sync)
		synchronize<dev>();
	else if(m_mem.template get<SyncedDevPtr<dev>>().ptr == nullptr && m_size != 0u)
		m_mem.template get<SyncedDevPtr<dev>>().ptr = make_udevptr_array<dev, char>(m_size);
	// [Weird] using the following two lines as a one-liner causes an internal compiler bug.
	auto pMem = m_mem.template get<SyncedDevPtr<dev>>().ptr.get();
	return pMem;
}
template < Device dev >
ConstArrayDevHandle_t<dev, char> GenericResource::acquire_const(bool sync) {
	if(sync)
		synchronize<dev>();
	else if(m_mem.template get<SyncedDevPtr<dev>>().ptr == nullptr && m_size != 0u)
		m_mem.template get<SyncedDevPtr<dev>>().ptr = make_udevptr_array<dev, char>(m_size);
	auto pMem = m_mem.template get<SyncedDevPtr<dev>>().ptr.get();
	return pMem;
}
template < Device dev >
void GenericResource::unload() {
	m_mem.template get<SyncedDevPtr<dev>>().ptr = nullptr;
	m_mem.template get<SyncedDevPtr<dev>>().synced = false;
}
template < Device dev >
bool GenericResource::is_resident() const noexcept {
	return m_mem.template get<SyncedDevPtr<dev>>().ptr != nullptr;
}

// Explicit instanciations
template struct DeviceManagerConcept<GenericResource>;

template void GenericResource::synchronize<Device::CPU>();
template void GenericResource::synchronize<Device::CUDA>();
template void GenericResource::synchronize<Device::OPENGL>();
template ArrayDevHandle_t<Device::CPU, char> GenericResource::acquire<Device::CPU>(bool);
template ArrayDevHandle_t<Device::CUDA, char> GenericResource::acquire<Device::CUDA>(bool);
template ArrayDevHandle_t<Device::OPENGL, char> GenericResource::acquire<Device::OPENGL>(bool);
template ConstArrayDevHandle_t<Device::CPU, char> GenericResource::acquire_const<Device::CPU>(bool);
template ConstArrayDevHandle_t<Device::CUDA, char> GenericResource::acquire_const<Device::CUDA>(bool);
template ConstArrayDevHandle_t<Device::OPENGL, char> GenericResource::acquire_const<Device::OPENGL>(bool);
template void GenericResource::unload<Device::CPU>();
template void GenericResource::unload<Device::CUDA>();
template void GenericResource::unload<Device::OPENGL>();
template bool GenericResource::is_resident<Device::CPU>() const noexcept;
template bool GenericResource::is_resident<Device::CUDA>() const noexcept;
template bool GenericResource::is_resident<Device::OPENGL>() const noexcept;

} // namespace mufflon
