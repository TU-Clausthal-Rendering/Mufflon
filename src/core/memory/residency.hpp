#pragma once

#include "export/dll_export.hpp"
#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <vector>

namespace mufflon { // There is no memory namespace on purpose

// Contains the possible data locations of the scene (logical devices).
enum class Device : unsigned char {
	CPU		= 1u,
	CUDA	= 2u,
	OPENGL	= 4u
};

// Generic type-trait for device-something-handles
template < Device dev, class H >
struct DeviceHandle {
	static constexpr Device DEVICE = dev;
	using HandleType = H;

	HandleType handle;
};

template < Device dev, class CH >
struct ConstDeviceHandle {
	static constexpr Device DEVICE = dev;
	using ConstHandleType = CH;

	ConstHandleType handle;
};

// Handle type exclusively for arrays, i.e. they must support vector-esque operations

template < Device dev, class T >
struct DeviceArrayHandle;
template < Device dev, class T >
struct ConstDeviceArrayHandle;

template < class T >
struct DeviceArrayHandle<Device::CPU, T> :
	public DeviceHandle<Device::CPU, T*> {
	using Type = T;
	using ValueType = T*;

	DeviceArrayHandle(ValueType* hdl) :
		DeviceHandle<Device::CPU, ValueType*>{ hdl }
	{}
};
template < class T >
struct ConstDeviceArrayHandle<Device::CPU, T> :
	public ConstDeviceHandle<Device::CPU, const T*> {
	using Type = T;
	using ValueType = T*;

	ConstDeviceArrayHandle(DeviceArrayHandle<Device::CPU, T> hdl) :
		ConstDeviceHandle<Device::CPU, const ValueType*>{ hdl.handle } {}
	ConstDeviceArrayHandle(const ValueType* hdl) :
		ConstDeviceHandle<Device::CPU, const ValueType*>{ hdl }
	{}
};

// TODO: what's wrong with device vector?
template < class T >
struct DeviceArrayHandle<Device::CUDA, T> :
	public DeviceHandle<Device::CUDA, T*> {
	using Type = T;
	using ValueType = T*;

	DeviceArrayHandle(ValueType* hdl) :
		DeviceHandle<Device::CUDA, ValueType*>{ hdl } {}
};

template < class T >
struct ConstDeviceArrayHandle<Device::CUDA, T> :
	public ConstDeviceHandle<Device::CUDA, const T*> {
	using Type = T;
	using ValueType = T*;

	ConstDeviceArrayHandle(DeviceArrayHandle<Device::CUDA, T> hdl) :
		ConstDeviceHandle<Device::CUDA, const ValueType*>{ hdl.handle } {}
	ConstDeviceArrayHandle(const ValueType* hdl) :
		ConstDeviceHandle<Device::CUDA, const ValueType*>{ hdl } {}
};

// Functions for synchronizing between array handles
template < class T >
void synchronize(ConstDeviceArrayHandle<Device::CPU, T> changed,
				 DeviceArrayHandle<Device::CUDA, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		cudaMalloc<T>(&sync, sizeof(T) * length);
	}
	cudaMemcpy(sync.handle, changed.handle, cudaMemcpyHostToDevice);
}
template < class T >
void synchronize(ConstDeviceArrayHandle<Device::CUDA, T> changed,
				 DeviceArrayHandle<Device::CPU, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		sync.handle = new T[length];
	}
	cudaMemcpy(sync.handle, changed.handle, cudaMemcpyDeviceToHost);
}

// Functions for unloading a handle from the device
template < class T >
void unload(DeviceArrayHandle<Device::CPU, T>& hdl) {
	delete[] hdl.handle;
	hdl.handle = nullptr;
}
template < class T >
void unload(DeviceArrayHandle<Device::CUDA, T>& hdl) {
	if(hdl.handle != nullptr) {
		cudaFree(hdl.handle);
		hdl.handle = nullptr;
	}
}

} // namespace mufflon::scene
