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

// Generic type for device-something-handles
template < Device dev, class T, class V >
struct DeviceHandle {
	static constexpr Device DEVICE = dev;
	using Type = T;
	using ValueType = V;
	using HandleType = ValueType*;
};

// Handle type exclusively for arrays, i.e. they must support vector-esque operations
template < Device dev, class T >
struct DeviceArrayHandle;

template < class T >
struct DeviceArrayHandle<Device::CPU, T> :
	public DeviceHandle<Device::CPU, T, std::vector<T>> {};

template < class T >
struct DeviceArrayHandle<Device::CUDA, T> :
	public DeviceHandle<Device::CUDA, T, thrust::device_vector<T>> {};

// Operations on the device arrays (override if they differ for some devices)
template < Device dev, template <Device, class> class H >
struct DeviceArrayOps {
	template < class T >
	using HandleType = H<dev, T>;
	using Type = typename HandleType::Type;
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
	static void copy(const typename H<other, T>::ValueType& changed,
					 ValueType& sync) {
		// TODO
		std::runtime_error("Copy between devices not yet supported!");
		//std::copy(changed.handle.cbegin(), changed.handle.cend(), sync.handle.begin());
	}
};

}} // namespace mufflon::scene
