#pragma once

#include <cstddef>
#include "residency.hpp"
#include "unique_device_ptr.h"
#include "util/tagged_tuple.hpp"

namespace mufflon {

/*
 * A bundel of pointers to memory blocks which may reside on different devices.
 */
class GenericResource {
public:
	GenericResource() : m_size(0) {}
	GenericResource(std::size_t size) : m_size(size) {}

	/* 
	 * Set the size for future allocations. All internal memories will be
	 * discarded, if they have a different size.
	 * size: size in bytes which should be consumed.
	 */
	void resize(std::size_t size);

	std::size_t size() const {
		return m_size;
	}

	/*
	 * Get a typized pointer to the memory on one device. If the memory does not
	 * exsist it will be created by the call.
	 * The const version will break instead (assertion).
	 */
	//template < Device dev >
	//char* acquire(bool sync = true) {
	//	if(sync)
	//		synchronize<dev>();
	//	else if(m_mem.template get<SyncedDevPtr<dev>>() == nullptr && m_size != 0u)
	//		m_mem.template get<SyncedDevPtr<dev>>() = make_udevptr_array<dev, char>(m_size);
	//	// [Weird] using the following two lines as a one-liner causes an internal compiler bug.
	//	auto* pMem = m_mem.template get<SyncedDevPtr<dev>>().get();
	//	return pMem;
	//}
	//template < Device dev >
	//const char* acquire_const(bool sync = true) {
	//	if(sync)
	//		synchronize<dev>();
	//	else if(m_mem.template get<SyncedDevPtr<dev>>() == nullptr && m_size != 0u)
	//		m_mem.template get<SyncedDevPtr<dev>>() = make_udevptr_array<dev, char>(m_size);
	//	auto* pMem = m_mem.template get<SyncedDevPtr<dev>>().get();
	//	return pMem;
	//}
	template < Device dev >
	ArrayDevHandle_t<dev, char> acquire(bool sync = true);
	template < Device dev >
	ConstArrayDevHandle_t<dev, char> acquire_const(bool sync = true);

	// Template variant for the synchronization
	template < Device dstDev >
	void synchronize();

	template < Device dev >
	void unload();

	void mark_changed(Device changed) noexcept;

	template < Device dev >
	bool is_resident() const noexcept;
private:
	template < Device dev >
	struct SyncedDevPtr {
		static constexpr Device DEVICE = dev;
		
		unique_device_ptr<dev, char[]> ptr;
		bool synced = false;
	};

	std::size_t m_size;
	util::TaggedTuple<
		SyncedDevPtr<Device::CPU>,
		SyncedDevPtr<Device::CUDA>,
		SyncedDevPtr<Device::OPENGL>> m_mem;
};

extern template void GenericResource::synchronize<Device::CPU>();
extern template void GenericResource::synchronize<Device::CUDA>();
extern template void GenericResource::synchronize<Device::OPENGL>();
extern template ArrayDevHandle_t<Device::CPU, char> GenericResource::acquire<Device::CPU>(bool);
extern template ArrayDevHandle_t<Device::CUDA, char> GenericResource::acquire<Device::CUDA>(bool);
extern template ArrayDevHandle_t<Device::OPENGL, char> GenericResource::acquire<Device::OPENGL>(bool);
extern template ConstArrayDevHandle_t<Device::CPU, char> GenericResource::acquire_const<Device::CPU>(bool);
extern template ConstArrayDevHandle_t<Device::CUDA, char> GenericResource::acquire_const<Device::CUDA>(bool);
extern template ConstArrayDevHandle_t<Device::OPENGL, char> GenericResource::acquire_const<Device::OPENGL>(bool);
extern template void GenericResource::unload<Device::CPU>();
extern template void GenericResource::unload<Device::CUDA>();
extern template void GenericResource::unload<Device::OPENGL>();
extern template bool GenericResource::is_resident<Device::CPU>() const noexcept;
extern template bool GenericResource::is_resident<Device::CUDA>() const noexcept;
extern template bool GenericResource::is_resident<Device::OPENGL>() const noexcept;

} // namespace mufflon
