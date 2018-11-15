#pragma once

#include "residency.hpp"
#include "util/flag.hpp"

namespace mufflon { // There is no memory namespace on purpose

/*
 * Accessor classes allow a semi-automatic synchronization of resources over all supported
 * devices. An accessor provides the handles for any kind of device.
 * This intermediate layer allows to track changes of resources automatically.
 * The Accessor should only be used if the resource is going to be changed. Once the
 * Accessor lifetime ends, the resource is flagged as dirty.
 * The ConstAccessor never flags anything as dirty (read only resource).
 *
 * Rules when working with (Const)Accessors:
 *	- after a change of a resource the handle may change as well
 *	  -> call aquire..() again and update your handles each iteration.
 *	- if handles can be expected to be non-changing, but resource content still changes
 *	  -> call synchronize..() again.
 * If the resource did not change the above calls will be relatively cheap.
 * If handles may change for different resource types. E.g. Object-attributes might
 * require reallocations, causing the handle to change, whereas textures have an
 * immutable size and only require synchronize()
 */

// Provides read-only access
// DH: The type-trait of the underlying accessed data. The handle types are different
//		for each type (defined by DeviceHandle and its inherited forms).
template < class DH >
class ConstAccessor {
public:
	using DeviceHandle = DH;
	using ConstHandleType = typename DH::ConstHandleType;
	static constexpr Device DEVICE = DH::DEVICE;

	ConstAccessor(ConstHandleType handle) :
		m_handle(handle) {}
	ConstAccessor(const ConstAccessor&) = default;
	ConstAccessor(ConstAccessor&&) = default;
	ConstAccessor& operator=(const ConstAccessor&) = default;
	ConstAccessor& operator=(ConstAccessor&&) = default;
	~ConstAccessor() = default;

	const ConstHandleType& operator*() const {
		return m_handle;
	}

	const ConstHandleType* operator->() const {
		return &m_handle;
	}

private:
	ConstHandleType m_handle;
};

// Provides read-and-write access to the attribute data. Flags as dirty upon destruction
template < class DH >
class Accessor {
public:
	using DeviceHandle = DH;
	using HandleType = typename DH::HandleType;
	static constexpr Device DEVICE = DH::DEVICE;

	Accessor(HandleType handle, util::DirtyFlags<Device>& flags) :
		m_handle(handle), m_flags(flags) {}
	Accessor(const Accessor&) = delete;
	Accessor(Accessor&&) = default;
	Accessor& operator=(const Accessor&) = delete;
	Accessor& operator=(Accessor&&) = default;
	~Accessor() {
		m_flags.mark_changed(DEVICE);
	}

	HandleType& operator*() {
		return m_handle;
	}

	HandleType* operator->() {
		return &m_handle;
	}

private:
	HandleType m_handle;
	util::DirtyFlags<Device>& m_flags;
};

} // namespace mufflon::scene