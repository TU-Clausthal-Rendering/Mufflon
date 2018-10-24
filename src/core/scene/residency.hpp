#pragma once

#include "util/flag.hpp"
#include "util/tagged_tuple.hpp"
#include <thrust/device_vector.h>
#include <algorithm>
#include <vector>

namespace mufflon { namespace scene {

// Contains the possible data locations of the scene.
enum class Residency : unsigned char {
	CPU		= 1u,
	CUDA	= 2u,
	OPENGL	= 4u
};


namespace handle_details {

// General class for specifying device array handle type
template < Residency dev, class T >
struct DeviceArrayHandleImpl;

template < class T >
struct DeviceArrayHandleImpl<Residency::CPU, T> {
	using HandleType = std::vector<T>;
};

template < class T >
struct DeviceArrayHandleImpl<Residency::CUDA, T> {
	using HandleType = thrust::device_vector<T>;
};

// TODO: OpenGL handle

} // namespace handle_details

// Actual device array handle
template < Residency dev, class T >
struct DeviceArrayHandle {
	static constexpr Residency DEVICE = dev;
	using Type = T;
	using HandleType = typename handle_details::DeviceArrayHandleImpl<dev, T>::HandleType;

	HandleType handle;
};

namespace handle_details {

// Helper struct for handle copy
template < Residency dev1, Residency dev2, class T >
struct ArrayCopyer {
	static void copy(DeviceArrayHandle<dev1, T>& changed, DeviceArrayHandle<dev1, T>& sync) {
		std::copy(changed.handle.cbegin(), changed.handle.cend(), sync.handle.begin());
	}
};

// Helper struct for device array sizes
template < Residency dev, class T >
struct ArraySize {
	static std::size_t size(DeviceArrayHandle<dev, T>& sync) {
		return sync.handle.size();
	}

	static void resize(DeviceArrayHandle<dev, T>& sync, std::size_t size) {
		sync.handle.resize(size);
	}
};

// TODO: different copy and size behavior for OpenGL!

} // namespace handle_details

// Attribute handles
template < class T, Residency... devs >
using DeviceArrayHandles = util::TaggedTuple<DeviceArrayHandle<devs, T>...>;

// Operations on the device arrays (override if they differ for some devices)
template < Residency dev, class T >
struct DeviceArrayOps {
	using HandleType = DeviceArrayHandle<dev, T>;

	static std::size_t get_size(const HandleType& sync) {
		return sync.handle.size();
	}

	static std::size_t get_capacity(const HandleType& sync) {
		return sync.handle.capacity();
	}
	
	static void resize(HandleType& sync, std::size_t elems) {
		return sync.handle.resize(elems);
	}

	static void reserve(HandleType& sync, std::size_t elems) {
		return sync.handle.reserve(elems);
	}

	static void clear(HandleType& sync) {
		sync.handle.clear();
	}

	template < Residency other >
	static void copy(const DeviceArrayHandle<other, T>& changed, HandleType& sync) {
		// TODO
		std::runtime_error("Copy between devices not yet supported!");
		//std::copy(changed.handle.cbegin(), changed.handle.cend(), sync.handle.begin());
	}
};

}} // namespace mufflon::scene
