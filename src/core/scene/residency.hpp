#pragma once

#include "export/dll_export.hpp"
#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <vector>

namespace mufflon { namespace scene {

// Contains the possible data locations of the scene (logical devices).
enum class Device : unsigned char {
	CPU		= 1u,
	CUDA	= 2u,
	OPENGL	= 4u
};

// Generic type-trait for device-something-handles
template < Device dev, class H, class CH = H >
struct DeviceHandle {
	static constexpr Device DEVICE = dev;
	using HandleType = H;
	using ConstHandleType = CH;

	HandleType handle;
};

// Handle type exclusively for arrays, i.e. they must support vector-esque operations

template < Device dev, class T >
struct DeviceArrayHandle;

template < class T >
struct DeviceArrayHandle<Device::CPU, T> :
	public DeviceHandle<Device::CPU, T*, const T*> {
	using Type = T;
	using ValueType = T*;

	DeviceArrayHandle(ValueType* hdl) :
		DeviceHandle<Device::CPU, ValueType*>{ hdl }
	{}
};

// TODO: what's wrong with device vector?
template < class T >
struct DeviceArrayHandle<Device::CUDA, T> :
	public DeviceHandle<Device::CUDA, T*, const T*> {
	using Type = T;
	using ValueType = T*;

	DeviceArrayHandle(ValueType* hdl) :
		DeviceHandle<Device::CUDA, ValueType*>{ hdl } {}
};

// Functions for synchronizing between array handles
template < class T >
void synchronize(const DeviceArrayHandle<Device::CPU, T>& changed,
				 DeviceArrayHandle<Device::CUDA, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		cudaMalloc<T>(&sync, sizeof(T) * length);
	}
	cudaMemcpy(sync.handle, changed.handle, cudaMemcpyHostToDevice);
}
template < class T >
void synchronize(const DeviceArrayHandle<Device::CUDA, T>& changed,
				 DeviceArrayHandle<Device::CPU, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		sync.handle = new T[length];
	}
	cudaMemcpy(sync.handle, changed.handle, cudaMemcpyDeviceToHost);
}

// Operations on the device arrays (override if they differ for some devices)
template < Device dev, class T, template <Device, class> class H >
struct DeviceArrayOps {
	using Type = T;
	using HandleType = H<dev, Type>;

	static std::size_t get_size(HandleType sync) {
		return sync.handle->size();
	}

	static std::size_t get_capacity(HandleType sync) {
		return sync.handle->capacity();
	}
	
	static void resize(HandleType sync, std::size_t elems) {
		return sync.handle->resize(elems);
	}

	static void reserve(HandleType sync, std::size_t elems) {
		return sync.handle->reserve(elems);
	}

	static void clear(HandleType sync) {
		sync.handle->clear();
	}

	template < Device other >
	static void copy(const H<other, Type> changed, HandleType sync) {
		// TODO
		std::runtime_error("Copy between devices not yet supported!");
		//std::copy(changed.handle.cbegin(), changed.handle.cend(), sync.handle.begin());
	}
};

}} // namespace mufflon::scene
