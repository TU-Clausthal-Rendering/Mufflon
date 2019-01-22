#pragma once

#include "allocator.hpp"
#include "dyntype_memory.hpp"
#include "util/tagged_tuple.hpp"
#include "util/assert.hpp"
#include "util/flag.hpp"
#include "core/memory/residency.hpp"
#include "core/concepts.hpp"

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
	template < Device dev >
	char* acquire(bool sync = true) {
		if(sync) synchronize<dev>();
		// [Weird] using the following two lines as a one-liner causes an internal compiler bug.
		auto* pMem = m_mem.template get<unique_device_ptr<dev, char[]>>().get();
		return pMem;
	}
	template < Device dev >
	const char* acquire_const(bool sync = true) {
		if(sync) synchronize<dev>();
		auto* pMem = m_mem.template get<unique_device_ptr<dev, char[]>>().get();
		return pMem;
	}

	// Template variant for the synchronization
	template < Device dstDev >
	void synchronize();

	template < Device dev >
	void unload() {
		m_mem.template get<unique_device_ptr<dev, char[]>>() = nullptr;
	}

	void mark_changed(Device changed) noexcept {
		m_dirty.mark_changed(changed);
	}

	void mark_synced(Device synced) noexcept {
		m_dirty.mark_synced(synced);
	}

	template < Device dev >
	bool is_resident() const noexcept {
		return m_mem.template get<unique_device_ptr<dev, char[]>>() != nullptr;
	}
private:
	std::size_t m_size;
	util::DirtyFlags<Device> m_dirty;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, char[]>,
		unique_device_ptr<Device::CUDA, char[]>> m_mem;
	//unique_device_ptr<Device::OPENGL, char> m_openglMem;
};
template struct DeviceManagerConcept<GenericResource>;

} // namespace mufflon
