#pragma once

#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include <thrust/device_vector.h>
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
template < Device dev, class H >
struct DeviceHandle {
	static constexpr Device DEVICE = dev;
	using HandleType = H;

	HandleType handle;
};

// Handle type exclusively for arrays, i.e. they must support vector-esque operations
template < Device dev, class T >
struct DeviceArrayHandle;

template < class T >
struct DeviceArrayHandle<Device::CPU, T> :
	public DeviceHandle<Device::CPU, std::vector<T>*> {
	using Type = T;
	using ValueType = std::vector<T>;
};

template < class T >
struct DeviceArrayHandle<Device::CUDA, T> :
	public DeviceHandle<Device::CUDA, thrust::device_vector<T>*> {
	using Type = T;
	using ValueType = thrust::device_vector<T>;
};

// Operations on the device arrays (override if they differ for some devices)
template < Device dev, class T, template <Device, class> class H >
struct DeviceArrayOps {
	using Type = T;
	using HandleType = H<dev, Type>;
	using ValueType = typename HandleType::ValueType;

	static std::size_t get_size(const ValueType& sync) {
		return sync.size();
	}

	static std::size_t get_capacity(const ValueType& sync) {
		return sync.capacity();
	}
	
	static void resize(ValueType& sync, std::size_t elems) {
		return sync.resize(elems);
	}

	static void reserve(ValueType& sync, std::size_t elems) {
		return sync.reserve(elems);
	}

	static void clear(ValueType& sync) {
		sync.clear();
	}

	template < Device other >
	static void copy(const typename H<other, Type>::ValueType& changed,
					 ValueType& sync) {
		// TODO
		std::runtime_error("Copy between devices not yet supported!");
		//std::copy(changed.handle.cbegin(), changed.handle.cend(), sync.handle.begin());
	}
};

}} // namespace mufflon::scene
