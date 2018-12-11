#pragma once

#include "allocator.hpp"
#include "dyntype_memory.hpp"
#include "util/tagged_tuple.hpp"
#include "util/assert.hpp"

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
	template < Device dev, typename T >
	T* acquire() {
		if(m_mem.template get<unique_device_ptr<dev, char>>() == nullptr)
			m_mem.template get<unique_device_ptr<dev, char>>() = make_udevptr_array<dev, char>(m_size);
		// [Weird] using the following two lines as a one-liner causes an internal compiler bug.
		auto* pMem = m_mem.template get<unique_device_ptr<dev, char>>().get();
		return as<T>(pMem);
	}
	template < Device dev, typename T >
	const T* acquireConst() const {
		auto* pMem = m_mem.template get<unique_device_ptr<dev, char>>().get();
		mAssert(pMem != nullptr);
		return as<T>(pMem);
	}

	// Template variant for the synchronization
	template < Device dstDev, Device srcDev >
	void synchronize();

	template < Device dev >
	void unload() {
		m_mem.template get<unique_device_ptr<dev, char>>() = nullptr;
	}
private:
	std::size_t m_size;
	util::TaggedTuple<
		unique_device_ptr<Device::CPU, char>,
		unique_device_ptr<Device::CUDA, char>> m_mem;
	//unique_device_ptr<Device::OPENGL, char> m_openglMem;
};

} // namespace mufflon
