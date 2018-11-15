#pragma once

namespace mufflon { // There is no memory namespace on purpose

// Contains the possible data locations of the scene (logical devices).
enum class Device : unsigned char {
	CPU		= 1u,
	CUDA	= 2u,
	OPENGL	= 4u
};

/*
 * Generic type-trait for device-something-handles.
 * The idea is to have custom types per device which can be set dependent on a
 * Device template parameter.
 * The DeviceHandle itself is only a raw helper to avoid wrong usage of the
 * actual types (no construction).
 * The intention is to provide type-traits ...DevHandle<> which inherit from DeviceHandle.
 * Those type-traits are used to map a Device to the specific type. Each custom type must
 * provide a HandleType and a ConstHandleType which can be fundamentally different.
 * Finally, there are type alias using the type-traits:
 * ...DevHandle_t<..>       = typename ...DevHandle<..>::HandleType;
 * Const...DevHandle_t<..>  = typename ...DevHandle<..>::ConstHandleType;
 */
template < Device dev >
struct DeviceHandle {
	static constexpr Device DEVICE = dev;
	DeviceHandle() = delete; // No instanciation (pure type trait).
};


// Handle type exclusively for arrays, i.e. they must support vector-esque operations
template < Device dev, class T >
struct ArrayDevHandle;

template < class T >
struct ArrayDevHandle<Device::CPU, T> : public DeviceHandle<Device::CPU> {
	using HandleType = T*;
	using ConstHandleType = const T*;
	using Type = T;
};

// TODO: what's wrong with device vector?
template < class T >
struct ArrayDevHandle<Device::CUDA, T> : public DeviceHandle<Device::CUDA> {
	using HandleType = T*;
	using ConstHandleType = const T*;
	using Type = T;
};

// Short type alias
template < Device dev, class T >
using ArrayDevHandle_t = typename ArrayDevHandle<dev, T>::HandleType;
template < Device dev, class T >
using ConstArrayDevHandle_t = typename ArrayDevHandle<dev, T>::ConstHandleType;

} // namespace mufflon
