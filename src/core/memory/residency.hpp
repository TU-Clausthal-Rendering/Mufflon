#pragma once
#include <cstdint>
#include "core/opengl/gl_buffer.hpp"

namespace mufflon { // There is no memory namespace on purpose

// Contains the possible data locations of the scene (logical devices).
enum class Device : unsigned char {
	NONE    = 0u,
	CPU		= 1u,
	CUDA	= 2u,
	OPENGL	= 4u
};

//static constexpr Device DeviceIterator[3] = {Device::CPU, Device::CUDA, Device::OPENGL};

// Convert to index access to be able to store things for different devices in arrays.
constexpr int NUM_DEVICES = 3;
template<Device dev>
inline constexpr int get_device_index() { return -1; }
template<>
inline constexpr int get_device_index<Device::CPU>() { return 0; }
template<>
inline constexpr int get_device_index<Device::CUDA>() { return 1; }
template<>
inline constexpr int get_device_index<Device::OPENGL>() { return 2; }

// Inline bits operators for Device.
inline Device operator&(Device a, Device b) {
	return Device(static_cast<int>(a) & static_cast<int>(b));
}

inline Device operator|(Device a, Device b) {
	return Device(static_cast<int>(a) | static_cast<int>(b));
}

inline Device operator^(Device a, Device b) {
	return Device(static_cast<int>(a) ^ static_cast<int>(b));
}

inline Device operator~(Device a) {
	return Device(~static_cast<int>(a));
}

// Conversion of a runtime parameter 'device' into a constexpr 'dev'.
// This is very useful to call templates via
// device_switch(device, foo<dev>());
// without rewriting the switch everytime.
// WARNING: this is a switch -> need to return or to break; at the end
// of 'expr'.
#define device_switch(device, expr)						\
	switch(device) {									\
		case Device::CPU: {								\
			constexpr Device dev = Device::CPU;			\
			expr;										\
		}												\
		case Device::CUDA: {							\
			constexpr Device dev = Device::CUDA;		\
			expr;										\
		}												\
		case Device::OPENGL: {							\
			constexpr Device dev = Device::OPENGL;		\
			expr;										\
		}												\
		default:										\
			mAssertMsg(false, "Unknown device type.");	\
	};

// Many code snippets are either CPU or CUDA. This can be detected at compile time.
// => no template parameter Device for algorithms.
#ifdef __CUDA_ARCH__
constexpr Device CURRENT_DEV = Device::CUDA;
#else
constexpr Device CURRENT_DEV = Device::CPU;
#endif

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

template < class T >
struct ArrayDevHandle<Device::OPENGL, T> : public DeviceHandle<Device::OPENGL> {
	using HandleType = gl::BufferHandle<T>;
	using ConstHandleType = gl::BufferHandle<T>;
	using Type = T;
};

// Short type alias
//template < Device dev, class T >
//using ArrayDevType_t = typename ArrayDevHandle<dev, T>::Type;
template < Device dev, class T >
using ArrayDevHandle_t = typename ArrayDevHandle<dev, T>::HandleType;
template < Device dev, class T >
using ConstArrayDevHandle_t = typename ArrayDevHandle<dev, T>::ConstHandleType;

} // namespace mufflon
