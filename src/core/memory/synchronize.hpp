#pragma once

#include "residency.hpp"
#include "core/cuda/error.hpp"
#include "util/flag.hpp"

namespace mufflon { // There is no memory namespace on purpose

namespace synchronize_detail {

template < std::size_t I, Device dev, class Tuple, class T, class... Args >
void synchronize_impl(Tuple& tuple, util::DirtyFlags<Device>& flags,
					  T& sync, Args... args) {
#ifdef __CUDACC__
	if(I < Tuple::size) {
#else // __CUDACC__
	if constexpr(I < Tuple::size) {
#endif // __CUDACC__
		// Workaround for VS2017 bug: otherwise you may use the 'Type' template of the
		// tagged tuple
		auto& changed = tuple.template get<I>();
		constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
		if(flags.has_changes(CHANGED_DEVICE)) {
			synchronize(changed, sync, std::forward<Args>(args)...);
		} else {
			synchronize_impl<I + 1u, dev>(tuple, flags, sync);
		}
	} else {
		(void)sync;
		(void)flags;
		(void)tuple;
	}
}

} // namespace synchronize_detail

// Synchronizes changes from the tuple to the given class
template < Device dev, class Tuple, class T, class... Args >
void synchronize(Tuple& tuple, util::DirtyFlags<Device>& flags, T& sync, Args... args) {
	if(flags.needs_sync(dev)) {
		if(flags.has_competing_changes())
			throw std::runtime_error("Competing changes for attribute detected!");
		// Synchronize
		synchronize_detail::synchronize_impl<0u, dev>(tuple, flags, sync,
													  std::forward<Args>(args)...);
	}
}


// Functions for synchronizing between array handles
template < class T >
void synchronize(ConstArrayDevHandle_t<Device::CPU, T> changed,
				 ArrayDevHandle_t<Device::CUDA, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		cuda::check_error(cudaMalloc<T>(&sync, sizeof(T) * length));
	}
	cuda::check_error(cudaMemcpy(sync.handle, changed.handle, cudaMemcpyDefault));
}
template < class T >
void synchronize(ConstArrayDevHandle_t<Device::CUDA, T> changed,
				 ArrayDevHandle_t<Device::CPU, T>& sync, std::size_t length) {
	if(sync.handle == nullptr) {
		sync.handle = new T[length];
	}
	cuda::check_error(cudaMemcpy(sync.handle, changed.handle, cudaMemcpyDefault));
}

// Functions for unloading a handle from the device
template < class T >
void unload(ArrayDevHandle_t<Device::CPU, T>& hdl) {
	delete[] hdl.handle;
	hdl.handle = nullptr;
}
template < class T >
void unload(ArrayDevHandle_t<Device::CUDA, T>& hdl) {
	if(hdl.handle != nullptr) {
		cuda::check_error(cudaFree(hdl.handle));
		hdl.handle = nullptr;
	}
}


// A number of copy primitives which call the internal required methods
// NOTE: There are synchronize() methods above which have a similar
// functionallity. However, they are too specialized (include allocations).
// MAYBE they can be replaced by this one
template < Device dev >
using DevPtr = void*;
template < Device dev >
using ConstDevPtr = const void*;
// TODO: OpenGL specialization
template < Device dstDev, Device srcDev >
inline void copy(DevPtr<dstDev> dst, ConstDevPtr<srcDev> src, std::size_t size ) {
	mAssertMsg(false, "Unimplemented copy specialization.");
}

template <>
inline void copy<Device::CPU, Device::CPU>(void* dst, const void* src, std::size_t size) {
	std::memcpy(dst, src, size);
}
template <>
inline void copy<Device::CUDA, Device::CPU>(void* dst, const void* src, std::size_t size) {
	cuda::check_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}
template <>
inline void copy<Device::CPU, Device::CUDA>(void* dst, const void* src, std::size_t size) {
	cuda::check_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}
template <>
inline void copy<Device::CUDA, Device::CUDA>(void* dst, const void* src, std::size_t size) {
	cuda::check_error(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}
// TODO: OpenGL (glBufferSubData with offset and object handle as target/src types

} // namespace mufflon